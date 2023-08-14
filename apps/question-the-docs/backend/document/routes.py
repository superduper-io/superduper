from fastapi import APIRouter, Body, Request
from superduperdb import superduper
from superduperdb.db.mongodb.query import Collection

from backend.document.models import Query
from backend.config import settings

document_router = APIRouter(prefix="/document", tags=["docs"])


def concept_assist_prompt_build(famous_person):
    return (
        f'Use the following description and code-snippets aboout SuperDuperDB to answer this question about SuperDuperDB in the voice of {famous_person}\n'
        'Do not use any other information you might have learned about other python packages\n'
        'Only base your answer on the code-snippets retrieved\n'
        '{context}\n\n'
        'Here\'s the question:\n'
    )


@document_router.post(
    "/query",
    response_description="Query document database for data to answer prompt",
)
def query_docs(request: Request, query: Query = Body(...)):
    db = superduper(request.app.mongodb_client.my_database_name)

    context_select = Collection(name="markdown").like({"text": query.query}, n=settings.NEAREST_TO_QUERY, vector_index="documentation_index").find({})
    response, _ = db.predict('gpt-3.5-turbo', input=query.query, context_select=context_select, context_key='text')
    return response.unpack()
