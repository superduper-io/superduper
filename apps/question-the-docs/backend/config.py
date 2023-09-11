import typing as t

try:
    from pydantic.v1 import BaseSettings
except ImportError:
    from pydantic import BaseSettings

PROMPT_LINE = '\
NOTE: Try to answer as much in line with the context as possible. \
Only provide an answer if you think the provided context enables you to \
formulate a sufficient answer. \
If the provided context is not sufficient or irrelevant to the query, please respond \
with "I have no sufficient answer based on the information available. Sorry.", \
if the query is like `hello`, `hi`, `how are you`, etc. please respond to it.'

PROMPT = f"""\
Given the following context {{context}},
please try to answer the question given below.

{PROMPT_LINE}

Here's the question:
"""


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
    documentation_sources: t.Sequence[str] = (
        'langchain',
        'superduperdb',
        'huggingface',
    )
    repo_config_path = '.default_repo_config.json'

    # Query configuration
    nearest_to_query: int = 5

    prompt: str = PROMPT


settings = AISettings()
