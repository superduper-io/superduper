import pandas as pd
import pymongo
from lancedb.context import contextualize
from superduperdb import superduper
from superduperdb.container.document import Document as D
from superduperdb.container.listener import Listener
from superduperdb.container.vector_index import VectorIndex
from superduperdb.db.mongodb.query import Collection
from superduperdb.ext.openai.model import OpenAI, OpenAIEmbedding


class ConceptAssisPrompt:
    def build(self, query, context):
        return f"Following is a Query and some context around the query, please try to give an answer for the query with the given context\nQuery: {query}\nContext: {context}"


class CodeAssistPrompt:
    def build(self, code, context):
        return f"Following is a code snippet which failed to work and some context around the code snippet, please try to fix it \nCode: {code}\nContext: {context}"


class SuperDuperDocumentationBackend:
    def __init__(self, db):
        # TODO: remove below line: only for testing
        db = pymongo.MongoClient(
            "mongodb://testmongodbuser:testmongodbpassword@localhost:27018"
        ).test

        self.db = superduper(db)
        self.chatbot = OpenAI(model="davinci")

    def setup(self):
        # TODO: Extract data from csv file.
        data = ["This is a test document.", "This is another test document."] * 100
        columns = ["text"]
        df = pd.DataFrame(data=data, columns=columns)

        df = contextualize(df).text_col("text").window(2).stride(2).to_df()

        documents = [D(data) for data in df.to_dict()]

        for batch_docs in range(0, len(documents), 10):
            self.db.execute(Collection("documentation_docs").insert_many(batch_docs))

        self.db.add(
            VectorIndex(
                identifier="documentation_index",
                indexing_listener=Listener(
                    model=OpenAIEmbedding(model="text-embedding-ada-002"),
                    key="text",
                    select=Collection(name="documentation_docs").find(),
                ),
            )
        )

    def contextualize(self, query: str):
        context_curr = self.db.execute(
            Collection(name="documentation_docs")
            .like({"text": query}, n=8, vector_index="documentation_index")
            .find({})
        )
        context = [c.unpack() for c in context_curr]
        context = "\n".join([c["text"] for c in context])
        return context

    def concept_completion(self, query):
        context = self.contextualize(query)
        prompt = ConceptAssisPrompt().build(query, context)
        return self.chatbot.predict(prompt)
