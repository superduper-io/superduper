from backend.ai.utils.github import save_github_md_files_locally
from backend.ai.utils.text import chunk_file_contents
from backend.config import settings

from superduperdb.container.document import Document
from superduperdb.db.mongodb.query import Collection


def load_ai_artifacts(db):
    artifacts = _create_ai_text_artifacts()
    documents = [Document({settings.vector_embedding_key: v}) for v in artifacts]
    db.execute(Collection(settings.mongo_collection_name).insert_many(documents))


def _create_ai_text_artifacts():
    filepaths = save_github_md_files_locally(
        settings.owner, settings.name, settings.documentation_location
    )
    ai_text_artifacts = chunk_file_contents(filepaths)
    return ai_text_artifacts
