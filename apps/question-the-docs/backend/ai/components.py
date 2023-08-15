import jinja2

from superduperdb.container.document import Document as D
from superduperdb.container.listener import Listener
from superduperdb.container.vector_index import VectorIndex
from superduperdb.db.mongodb.query import Collection
from superduperdb.ext.openai.model import OpenAIChatCompletion, OpenAIEmbedding

from backend.config import settings



def install_openai_chatbot(db):
    prompt = jinja2.Template(settings.PROMPT).render(settings=settings)
    db.add(
        OpenAIChatCompletion(
            takes_context=True,
            prompt=prompt,
            model=settings.QA_MODEL,
        )
    )


def install_openai_vector_index(db):
    db.add(
        VectorIndex(
            identifier=settings.VECTOR_INDEX_NAME,
            indexing_listener=Listener(
                model=OpenAIEmbedding(model=settings.VECTOR_EMBEDDING_MODEL),
                key=settings.VECTOR_EMBEDDING_KEY,
                select=Collection(name=settings.MONGO_COLLECTION_NAME).find(),
            ),
        )
    )


def setup_ai(db):
    install_openai_vector_index(db)
    install_openai_chatbot(db)
