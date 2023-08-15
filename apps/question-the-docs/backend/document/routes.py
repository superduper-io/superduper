from fastapi import APIRouter, Request
from superduperdb.db.mongodb.query import Collection

from backend.document.models import Query
from backend.config import settings

document_router = APIRouter(prefix="/document", tags=["docs"])


@document_router.post(
    "/query",
    response_description="Query document database for data to answer prompt",
)
def query_docs(request: Request, query: Query) -> dict:
    db = request.app.superduperdb
    collection = Collection(name=settings.MONGO_COLLECTION_NAME)
    # build your query here combining vector-search "like(...)"
    # with classical mongodb queries "find(...)"
    context_select = (
        collection
            .like(
                {"text": query.query},
                n=settings.nearest_to_query,
                vector_index="documentation_index"
            )
            .find()
    )
    response, _ = db.predict('gpt-3.5-turbo', input=query.query, context_select=context_select, context_key='text')
    return response.unpack()
