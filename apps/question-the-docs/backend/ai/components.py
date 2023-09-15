from backend.config import settings
from superduperdb.container.listener import Listener
from superduperdb.container.vector_index import VectorIndex
from superduperdb.db.mongodb.query import Collection
from superduperdb.ext.openai.model import OpenAIChatCompletion, OpenAIEmbedding


def install_ai_components(db):
    """
    Install the chatbot and vector index components into the database
    """
    db.add(_openai_chatbot())
    for src in settings.documentation_sources:
        db.add(_openai_vector_index(src))


def _openai_chatbot():
    return OpenAIChatCompletion(
        takes_context=True,
        prompt=settings.prompt,
        model=settings.qa_model,
    )


def _openai_vector_index(src):
    return VectorIndex(identifier=src, indexing_listener=_open_ai_listener(src))


def _open_ai_listener(src):
    return Listener(
        model=OpenAIEmbedding(model=settings.vector_embedding_model),
        key=settings.vector_embedding_key,
        select=Collection(name=src).find(),
        predict_kwargs={'chunk_size': 100},
    )
