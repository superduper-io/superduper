from enum import Enum

from pydantic import BaseModel


class Repo(str, Enum):
    superduperdb = 'superduperdb'
    langchain = 'langchain'
    fastchat = 'fastchat'


class Query(BaseModel):
    query: str
    collection_name: Repo


class Answer(BaseModel):
    answer: str
    source_urls: list
