from backend.config import settings
from backend.documents.models import Answer, Query
from fastapi import APIRouter, Request

from superduperdb.db.mongodb.query import Collection

documents_router = APIRouter(prefix='/documents', tags=['docs'])


@documents_router.post(
    '/query',
    response_description='Query document database for data to answer prompt',
)
async def query_docs(request: Request, query: Query) -> Answer:
    # Step 1: Build your query
    # Build your query here combining vector-search "like(...)"
    # with classical mongodb queries "find(...)"
    collection = Collection(name=query.collection_name)
    context_select = collection.like(
        {settings.vector_embedding_key: query.query},
        n=settings.nearest_to_query,
        vector_index=query.collection_name,
    ).find()
    db = request.app.superduperdb

    contexts = list(db.execute(context_select))
    src_urls = [context.unpack()['src_url'] for context in contexts]

    # Step 2: Execute your query
    db_response, _ = await db.apredict(
        'gpt-3.5-turbo',
        input=query.query,
        context_select=context_select,
        context_key=settings.vector_embedding_key,
    )
    return Answer(answer=db_response.unpack(), source_urls=src_urls)
