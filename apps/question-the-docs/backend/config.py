from pydantic import BaseSettings


class FastAPISettings(BaseSettings):
    mongo_uri: str = 'mongodb://localhost:27017/'
    mongo_db_name: str = 'documentation'
    mongo_collection_name: str = "docs"
    port: int = 8000
    host: str = "0.0.0.0"
    debug_mode: bool = False


class AISettings(FastAPISettings):
    # Model details
    vector_index_name: str = 'documentation_index'
    vector_embedding_model: str = 'text-embedding-ada-002'
    vector_embedding_key: str = 'text'
    qa_model: str = 'gpt-3.5-turbo'
    doc_file_levels: int = 3
    doc_file_ext: str = 'md'
    default_repos: list = [
        'https://github.com/SuperDuperDB/superduperdb/tree/main',
        'https://github.com/langchain-ai/langchain/tree/master',
        'https://github.com/lm-sys/FastChat/tree/main'
    ]

    # Query configuration
    nearest_to_query: int = 5


settings = AISettings()
