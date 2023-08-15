from pydantic import BaseSettings


class FastAPISettings(BaseSettings):
    mongo_uri: str
    mongo_db_name: str = 'documentation'
    mongo_collection_name: str = "docs"
    port: int = 8000
    host: str = "0.0.0.0"
    debug_mode: bool = False


class RepoSettings(FastAPISettings):
    owner: str = 'SuperDuperDB'
    name: str = 'superduperdb'
    documentation_location: str = 'docs'


class AISettings(RepoSettings):
    # Model details
    vector_index_name: str = 'documentation_index'
    vector_embedding_model: str = 'text-embedding-ada-002'
    vector_embedding_key: str = 'txt'
    qa_model: str = 'gpt-3.5-turbo'

    # Query configuration
    nearest_to_query: int = 5


settings = AISettings()
