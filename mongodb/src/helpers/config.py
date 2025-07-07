from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings (BaseSettings):
    
    APP_NAME: str
    APP_VERSION: str
    OPENAI_API_KEY: str
    
    FILE_ALLOWED_TYPES: list
    FILE_MAX_SIZE: int
    
    FILE_DEFAULT_CHUNK_SIZE: int
    
    MONGODB_URI: str
    MONGODB_DATABASE: str
    
    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str

    OPENAI_API_KEY: str = None
    OPENAI_API_URL: str = None
    COHERE_API_KEY: str = None
    GEMINI_API_KEY: str = None

    GENERATION_MODEL_ID: str = None
    EMBEDDING_MODEL_ID: str = None
    EMBEDDING_MODEL_SIZE: int = None
    
    DEFAULT_INPUT_MAX_CHARACTERS: int = None
    DEFAULT_GENERATION_MAX_OUTPUT_TOKENS: int = None
    DEFAULT_GENERATION_TEMPERATURE: float = None
    
    VECTOR_DB_BACKEND: str
    VECTOR_DB_PATH: str
    VECTOR_DB_DISTANCE_METHOD: str = None
    
    PRIMARY_LANGUAGE: str = "en"
    DEFAULT_LANGUAGE: str = "en"
    
    class Config:
        env_file = ".env"

def get_settings ():
    return Settings()
