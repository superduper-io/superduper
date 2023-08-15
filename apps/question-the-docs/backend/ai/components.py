'''
INSERT SUMMARY ON THIS MODULE HERE
'''

from backend.config import settings

from superduperdb.container.listener import Listener
from superduperdb.container.vector_index import VectorIndex
from superduperdb.db.mongodb.query import Collection
from superduperdb.ext.openai.model import OpenAIChatCompletion, OpenAIEmbedding

PROMPT = '''Use the following descriptions and code-snippets to answer the question.
Do NOT use any information you have learned about other python packages.
ONLY base your answer on the code-snippets retrieved:

{context}

Here's the question:
'''


def install_openai_chatbot(db):
    db.add(
        OpenAIChatCompletion(
            takes_context=True,
            prompt=PROMPT,
            model=settings.qa_model,
        )
    )


def install_openai_vector_index(db):
    db.add(
        VectorIndex(
            identifier=settings.vector_index_name,
            indexing_listener=Listener(
                model=OpenAIEmbedding(model=settings.vector_embedding_model),
                key=settings.vector_embedding_key,
                select=Collection(name=settings.mongo_collection_name).find(),
            ),
        )
    )


def install_ai_components(db):
    install_openai_vector_index(db)
    install_openai_chatbot(db)
