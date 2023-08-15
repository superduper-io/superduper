from pydantic import BaseSettings


PROMPT = '''Use the following descriptions and code-snippets aboout {{ settings.PROJECT_NAME }} to answer this question about SuperDuperDB
Do not use any other information you might have learned about other python packages
Only base your answer on the code-snippets retrieved:

{context}

Here's the question:
'''


class Settings(BaseSettings):
    # Mongo connection
    MONGO_URI: str = 'mongodb://localhost:27017'
    MONGO_DB_NAME: str = 'documentation'
    MONGO_COLLECTION_NAME: str = 'docs'

    # Code base to configure
    PATH_TO_REPO: str = './superduperdb'
    GIT_URL: str = 'git@github.com:SuperDuperDB/superduperdb.git'

    # Data configurations
    DOC_FILE_LEVELS: int = 3
    DOC_FILE_EXT: str = 'md'
    STRIDE: int = 5
    WINDOW_SIZE: int = 10

    # Model details
    VECTOR_INDEX_NAME: str = 'documentation_index'
    VECTOR_EMBEDDING_MODEL: str = 'text-embedding-ada-002'
    VECTOR_EMBEDDING_KEY: str = 'txt'
    QA_MODEL: str = 'gpt-3.5-turbo'
    PROMPT: str = PROMPT

    # FastAPI details
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    DEBUG_MODE: bool = False
    
    # Query configurations
    NEAREST_TO_QUERY: int = 5



settings = Settings()
