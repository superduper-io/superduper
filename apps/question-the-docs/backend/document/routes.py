import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, Request
from lancedb.context import contextualize
from superduperdb import superduper
from superduperdb.container.document import Document as D
from superduperdb.container.listener import Listener
from superduperdb.container.vector_index import VectorIndex
from superduperdb.db.mongodb.query import Collection
from superduperdb.ext.openai.model import OpenAIChatCompletion, OpenAIEmbedding

from backend.document.models import Query

document_router = APIRouter(prefix="/document", tags=["docs"])


def concept_assist_prompt_build(famous_person, query, context):
    return (
        "Following is a Query and some context around the query, please "
        "try to give an answer for the query with the given context and "
        f"please respond in the voice of {famous_person}\nQuery: "
        f"{query}\nContext: {context}"
    )


@document_router.post(
    "/query",
    response_description="Query document database for data to answer prompt",
)
def query_docs(request: Request, query: Query = Body(...)):
    db = superduper(request.app.mongodb_client.my_database_name)

    context_curr = db.execute(
        Collection(name="markdown")
        .like({"text": query.query}, n=8, vector_index="documentation_index")
        .find({})
    )
    context = [c.unpack() for c in context_curr]
    context = "\n".join([c["text"] for c in context])
    prompt = concept_assist_prompt_build("The Terminator", query, context)

    return OpenAIChatCompletion(model="gpt-3.5-turbo").predict(prompt, one=True)


@document_router.post(
    "/populate",
    response_description="Populate the database with documentation",
)
def populate(request: Request):
    db = superduper(request.app.mongodb_client.my_database_name)
    # TODO: Replace data with user docs

    raw_df = pd.read_csv("backend/data/doc-words/source.csv")

    raw_df["text"] = raw_df["text"].astype(str)
    # raw_df["text"].replace("", np.nan, inplace=True)
    raw_df["text"].fillna(" ", inplace=True)

    df = contextualize(raw_df).text_col("text").window(60).stride(17).to_df()

    documents = [D({"text": v}) for v in df["text"].values]

    db.execute(Collection("markdown").insert_many(documents))

    db.add(
        VectorIndex(
            identifier="documentation_index",
            indexing_listener=Listener(
                model=OpenAIEmbedding(model="text-embedding-ada-002"),
                key="text",
                select=Collection(name="markdown").find(),
            ),
        )
    )

    # TODO: Add correct response for user
