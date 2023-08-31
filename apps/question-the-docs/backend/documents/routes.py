import asyncio

import async_timeout
from backend.config import settings
from backend.documents.models import Query, Source
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from superduperdb.db.mongodb.query import Collection

GENERATION_TIMEOUT_SEC = 30

documents_router = APIRouter(prefix='/documents/vector-search', tags=['docs'])


@documents_router.post(
    '',
    response_description='Return most relevant document URLs for query',
)
async def vector_search(request: Request, query: Query) -> Source:
    # Step 1: Build your query
    # Build your vector database query by combining vector-search
    # "like(...)" with a standard mongodb query "find(...)"
    collection = Collection(name=query.collection_name)
    context_select = collection.like(
        {settings.vector_embedding_key: query.query},
        n=settings.nearest_to_query,
        vector_index=query.collection_name,
    ).find()

    # Step 2: Execute your query!
    db = request.app.superduperdb
    contexts = list(db.execute(context_select))

    return Source(urls=[context.unpack()['src_url'] for context in contexts])


@documents_router.post(
    '/summary',
    response_description='Return summary of most relevant document URLs for query',
)
async def query_docs(request: Request, query: Query) -> StreamingResponse:
    # Step 1: Build your query
    # Build your vector database query by combining vector-search
    # "like(...)" with a standard mongodb query "find(...)"
    collection = Collection(name=query.collection_name)
    context_select = collection.like(
        {settings.vector_embedding_key: query.query},
        n=settings.nearest_to_query,
        vector_index=query.collection_name,
    ).find()

    # Step 2: Execute your query!
    # This step executes the vector search query as in the previous example,
    # and then submits the results to the OpenAI Chat endpoint for summarization
    db = request.app.superduperdb
    db_response, _ = await db.apredict(
        'gpt-3.5-turbo',
        input=query.query,
        context_select=context_select,
        context_key=settings.vector_embedding_key,
        stream=True,
    )

    return StreamingResponse(
        stream_openai_content(db_response.content), media_type='text/event-stream'
    )


async def stream_openai_content(content):
    "Helper function for streaming content from the OpenAI Chat endpoint"
    async with async_timeout.timeout(GENERATION_TIMEOUT_SEC):
        try:
            async for chunk in content:
                if 'content' in chunk['choices'][0]['delta']:
                    yield chunk['choices'][0]['delta']['content']
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Stream timed out")
