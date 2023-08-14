from pydantic import BaseModel, Field


class Query(BaseModel):
    query: str = Field(...)
