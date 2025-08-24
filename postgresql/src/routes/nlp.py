from fastapi import APIRouter, status, Request
from fastapi.responses import JSONResponse
from .schemas.nlp_schema import PushRequestSchema, SearchRequestSchema
from models.ProjectModel import ProjectModel
from controllers import NLPController
from models import ResponseSignal
from tasks.data_indexing import index_data_content
import logging

logger = logging.getLogger('uvicorn.error')

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1", "nlp"]
)

@nlp_router.post("/index/push/{project_id}")
async def index_project(request: Request, project_id: int, push_request: PushRequestSchema):
    
    task = index_data_content.delay(
        project_id=project_id,
        do_reset=push_request.do_reset
    )
    
    return JSONResponse(
        content={
            "Signal": ResponseSignal.DATA_PUSH_TASK_READY.value,
            "task_id": task.id
        }
    )


@nlp_router.get("/index/info/{project_id}")
async def get_project_index_info (request: Request, project_id: int):
    
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    
    project = await project_model.get_project_or_create_one(project_id=project_id)
    
    nlp_controller = NLPController(
        vector_db_client=request.app.vector_db_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser
    )
    
    collection_info = await nlp_controller.get_vector_db_collection_info(project=project)
    
    return JSONResponse(
        content={
            "Signal": ResponseSignal.GET_VECTOR_DB_COLLECTION_INFO_SUCCESS.value,
            "CollectionInfo": collection_info,
        }
    )


@nlp_router.post("/index/search/{project_id}")
async def search_index (request: Request, project_id: int, search_request: SearchRequestSchema):
    
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    
    project = await project_model.get_project_or_create_one(project_id=project_id)
    
    if not project:
        logger.error(f"Project with ID {project_id} not found.")
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "Signal": ResponseSignal.PROJECT_NOT_FOUND.value
            }
        )
    
    nlp_controller = NLPController(
        vector_db_client=request.app.vector_db_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser
    )
    
    search_results = await nlp_controller.search_vector_db_collection(
        project=project,
        text=search_request.text,
        limit=search_request.limit
    )
    
    if not search_results:
        logger.error(f"No search results found for project {project_id} with query '{search_request.text}'.")
        
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "Signal": ResponseSignal.VECTOR_SEARCH_FAILED.value
            }
        )
    
    return JSONResponse(
        content={
            "Signal": ResponseSignal.VECTOR_SEARCH_SUCCESS.value,
            "SearchResults": [ result.dict() for result in search_results ],
        }
    )

@nlp_router.post("/index/answer/{project_id}")
async def answer_rag (request: Request, project_id: int, search_request: SearchRequestSchema):
    
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    
    project = await project_model.get_project_or_create_one(project_id=project_id)
    
    if not project:
        logger.error(f"Project with ID {project_id} not found.")
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "Signal": ResponseSignal.PROJECT_NOT_FOUND.value
            }
        )
    
    nlp_controller = NLPController(
        vector_db_client=request.app.vector_db_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser
    )
    
    answer, full_prompt, chat_history = await nlp_controller.answer_rag_question(
        project=project,
        query=search_request.text,
        limit=search_request.limit
    )
    
    if not answer:
        logger.error(f"Failed to answer question for project {project_id} with query '{search_request.text}'.")
        
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "Signal": ResponseSignal.RAG_ANSWER_FAILED.value
            }
        )
    
    return JSONResponse(
        content={
            "Signal": ResponseSignal.RAG_ANSWER_SUCCESS.value,
            "Answer": answer,
            "FullPrompt": full_prompt,
            "ChatHistory": chat_history
        }
    )

