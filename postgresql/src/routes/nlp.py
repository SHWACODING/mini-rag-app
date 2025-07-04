from fastapi import FastAPI, APIRouter, status, Request
from fastapi.responses import JSONResponse
from .schemas.nlp_schema import PushRequestSchema, SearchRequestSchema
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from controllers import NLPController
from models import ResponseSignal
import logging
from tqdm.auto import tqdm

logger = logging.getLogger('uvicorn.error')

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1", "nlp"]
)

@nlp_router.post("/index/push/{project_id}")
async def index_project(request: Request, project_id: int, push_request: PushRequestSchema):
    
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    
    chunk_model = await ChunkModel.create_instance(db_client=request.app.db_client)
    
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
    
    has_records = True
    page_num = 1
    inserted_item_count = 0
    idx = 0
    
    # Create Collection if Not Exists
    collection_name = nlp_controller.create_collection_name(project_id=project.project_id)
    
    _ = await request.app.vector_db_client.create_collection(
        collection_name=collection_name,
        embedding_size=request.app.embedding_client.embedding_size,
        do_reset=push_request.do_reset
    )
    
    # Setup Batching
    total_chunks_count = await chunk_model.get_total_chunks_count(project_id=project.project_id)
    
    pbar = tqdm(
        total=total_chunks_count,
        desc=f"Vector Indexing Project For {project.project_id} Chunks",
        position=0,
    )
    
    while has_records:
        page_chunks = await chunk_model.get_poject_chunks(project_id=project.project_id, page_num=page_num)
        
        if len(page_chunks):
            page_num += 1
        
        if not page_chunks or len(page_chunks) == 0:
            has_records = False
            break
        
        chunks_ids = [ c.chunk_id for c in page_chunks ]
        idx += len(page_chunks)
        
        is_inserted = await nlp_controller.index_into_vector_db(
            project=project,
            chunks=page_chunks,
            chunks_ids=chunks_ids
        )
        
        if not is_inserted:
            logger.error(f"Failed to index chunks for project {project_id}.")
            
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "Signal": ResponseSignal.INSERT_INTO_VECTOR_DB_FAILED.value
                }
            )
        
        pbar.update(len(page_chunks))
        
        inserted_item_count += len(page_chunks)
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "Signal": ResponseSignal.INSERT_INTO_VECTOR_DB_SUCCESS.value,
            "InsertedItemCount": inserted_item_count,
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

