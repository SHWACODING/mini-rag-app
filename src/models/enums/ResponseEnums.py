from enum import Enum

class ResponseSignal (Enum):
    
    FILE_VALIDATED_SUCCESSFULLY="File Validated Successfully!!"
    FILE_TYPE_NOT_SUPPORTED="File Type Not Supported!!"
    FILE_SIZE_EXCEEDED="File Size Exceeded!!"
    FILE_UPLOADED_SUCCESSFULLY="Uploaded Successfully!!"
    FILE_FAILED_UPLOAD="Failed To Upload!!"
    PROCESSING_SUCCESSEEDED="Processing Successeded!!"
    PROCESSING_FAILED="Processing Failed!!"
    NO_FILES_FOUND="No Files Found!!"
    FILE_ID_ERROR="No Such Files With That ID!!"
    PROJECT_NOT_FOUND="Project Not Found!!"
    INSERT_INTO_VECTOR_DB_FAILED="Failed To Insert Into Vector DB!!"
    INSERT_INTO_VECTOR_DB_SUCCESS="Inserted Into Vector DB Successfully!!"
    GET_VECTOR_DB_COLLECTION_INFO_SUCCESS="Got Vector DB Collection Info Successfully!!"
    GET_VECTOR_DB_COLLECTION_INFO_FAILED="Failed To Get Vector DB Collection Info!!"
    VECTOR_SEARCH_SUCCESS="Vector Search Success!!"
    VECTOR_SEARCH_FAILED="Vector Search Failed!!"
    RAG_ANSWER_FAILED="RAG Answer Failed!!"
    RAG_ANSWER_SUCCESS="RAG Answer Success!!"
