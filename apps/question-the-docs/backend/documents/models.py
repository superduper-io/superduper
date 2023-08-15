from pydantic import BaseModel, Field


class Query(BaseModel):
    query: str = Field(...)


class Answer(BaseModel):
    answer: str = Field(...)
