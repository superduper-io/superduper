from pydantic import BaseModel, Field


class Query(BaseModel):
    query: str = Field(...)
    document_index: str = Field(...)


class Answer(BaseModel):
    answer: str = Field(...)
