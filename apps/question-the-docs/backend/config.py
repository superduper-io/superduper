import typing as t

from pydantic import BaseSettings


class FastAPISettings(BaseSettings):
    mongo_uri: str = 'mongodb://localhost:27017/'
    mongo_db_name: str = 'documentation'
    port: int = 8000
    host: str = '0.0.0.0'
    debug_mode: bool = False


class AISettings(FastAPISettings):
    # Model details
    vector_embedding_model: str = 'text-embedding-ada-002'
    vector_embedding_key: str = 'text'
    qa_model: str = 'gpt-3.5-turbo'
    default_repos: t.List[str] = [
        'superduperdb',
        'langchain',
        'fastchat',
    ]

    # Query configuration
    nearest_to_query: int = 5

    prompt: str = '''Use the following descriptions and code-snippets to answer the question.
    Do NOT use any information you have learned about other python packages.
    ONLY base your answer on the code-snippets retrieved:
    
    {context}
    
    Here's the question:
    '''


settings = AISettings()
