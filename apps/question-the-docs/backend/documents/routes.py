import typing as t

from backend.ai.utils.github import repos
from backend.config import settings
from backend.documents.models import Answer, Query
from fastapi import APIRouter, Request

from superduperdb.db.mongodb.query import Collection

documents_router = APIRouter(prefix='/documents', tags=['docs'])


@documents_router.get('/documentation_list')
async def documentation_list() -> t.Sequence[str]:
    names = []
    for _, config in repos().items():
        names.append(config.documentation_name)
    return names


@documents_router.post(
    '/query',
    response_description='Query document database for data to answer prompt',
)
async def query_docs(request: Request, query: Query) -> Answer:
    #
    # Step 1: Build your query
    #
    # Build your query here combining vector-search "like(...)"
    # with classical mongodb queries "find(...)"
    #
    collection = Collection(name=query.collection_name)

    to_find = {settings.vector_embedding_key: query.query}
    context_select = collection.like(
        to_find,
        n=settings.nearest_to_query,
        vector_index=query.collection_name,
    ).find()

    db = request.app.superduperdb
    src_urls = {c.unpack()['src_url'] for c in db.execute(context_select)}

    # Step 2: Execute your query
    db_response, _ = await db.apredict(
        'gpt-3.5-turbo',
        input=query.query,
        context_select=context_select,
        context_key=settings.vector_embedding_key,
    )
    return Answer(answer=db_response.unpack(), source_urls=list(src_urls))
