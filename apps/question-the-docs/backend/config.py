from pydantic import BaseSettings


class Settings(BaseSettings):
    DB_URL: str
    DB_NAME: str
    DEBUG_MODE: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    PATH_TO_REPO = './'
    DOC_FILE_LEVELS = 3
    DOC_FILE_EXT = 'md'
    NEAREST_TO_QUERY = 5




settings = Settings()
