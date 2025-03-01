from enum import Enum

class ResponseSignal (Enum):
    
    FILE_VALIDATED_SUCCESSFULLY="File Validated Successfully!!"
    FILE_TYPE_NOT_SUPPORTED="File Type Not Supported!!"
    FILE_SIZE_EXCEEDED="File Size Exceeded!!"
    FILE_UPLOADED_SUCCESSFULLY="Uploaded Successfully!!"
    FILE_FAILED_UPLOAD="Failed To Upload!!"
