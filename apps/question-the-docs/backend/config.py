from pydantic import BaseSettings


class Settings(BaseSettings):
    DB_URL: str
    DB_NAME: str
    DEBUG_MODE: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000


settings = Settings()