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
        'langchain',
    ]

    # Query configuration
    nearest_to_query: int = 5

    prompt: str = '''
    Given the following context {context},
    please try to answer the question given below.

    NOTE: Try to answer as much in line with the context as possible. Only provide an answer if you think the provided context enables you to formulate a sufficient answer. If the provided context is not sufficient or irrelevant to the query, please respond with "I have no sufficient answer based on the information available. Sorry.", if the query is like `hello`, `hi`, `how are you`, etc. please repsond to it.

    Here's the question:
    '''


settings = AISettings()
