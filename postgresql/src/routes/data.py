from fastapi import FastAPI, APIRouter, Depends, UploadFile, status, Request
from fastapi.responses import JSONResponse
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController
from models import ResponseSignal
import os
import aiofiles
import logging
from .schemas.data_schema import ProcessRequest
from models.ProjectModel import ProjectModel
from models.AssetModel import AssetModel
from models.db_schemes import Asset
from models.enums.AssetTypeEnums import AssetTypeEnum
from tasks.file_processing import process_project_files

logger = logging.getLogger('uvicorn.error')

data_router = APIRouter (
    prefix="/api/v1/data",
    tags=["api_v1", "data"]
)

@data_router.post("/upload/{project_id}")
async def upload_data (request: Request, project_id: int, file: UploadFile, app_settings: Settings = Depends(get_settings)):
    
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    
    project = await project_model.get_project_or_create_one(project_id=project_id)
    
    # Validate The File Properties ??
    data_controller = DataController()
    
    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)
    
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "Signal": result_signal
            }
        )
    
    project_dir_path = ProjectController().get_project_path(project_id=project_id)
    
    file_path, file_id = data_controller.generate_unique_filepath(
        original_file_name=file.filename,
        project_id=project_id
    )
    
    try :
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    
    except Exception as e:
        
        logger.error(f"Error While Uploading That File : {e}")
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "Signal": ResponseSignal.FILE_FAILED_UPLOAD.value
            }
        )
    
    # Store The Assets into The Database
    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
    asset_resource = Asset (
        asset_project_id=project.project_id,
        asset_type=AssetTypeEnum.FILE.value,
        asset_name=file_id,
        asset_size=os.path.getsize(file_path)
    )
    
    asset_record = await asset_model.create_asset(asset=asset_resource)
    
    return JSONResponse(
        content={
            "Signal": ResponseSignal.FILE_UPLOADED_SUCCESSFULLY.value,
            "File_Id": str(asset_record.asset_id),
        }
    )

@data_router.post ("/process/{project_id}")
async def process_endpoint (request: Request, project_id: int, process_request: ProcessRequest):

    chunk_size = process_request.chunk_size
    overlap_size = process_request.overlap_size
    do_reset = process_request.do_reset
    
    task = process_project_files.delay(
        project_id=project_id,
        file_id=process_request.file_id,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        do_reset=do_reset
    )
    
    return JSONResponse(
        content={
            "Signal": ResponseSignal.PROCESSING_SUCCESSEEDED.value,
            "task_id": task.id
        }
    )