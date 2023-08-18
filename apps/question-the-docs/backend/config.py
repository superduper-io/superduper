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
    doc_file_ext: str = 'md'
    default_repos: dict = {
        'superduperdb': {
            'url': 'https://github.com/SuperDuperDB/superduperdb/tree/main',
            'documentation_url': 'https://superduperdb.github.io/superduperdb',
        },
        'langchain': {
            'url': 'https://github.com/langchain-ai/langchain/tree/master',
            'documentation_url': '',
        },
        'fastchat': {
            'url': 'https://github.com/lm-sys/FastChat/tree/main',
            'documentation_url': '',
        },
    }

    # Query configuration
    nearest_to_query: int = 5


settings = AISettings()
