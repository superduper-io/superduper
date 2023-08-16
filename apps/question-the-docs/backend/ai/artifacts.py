from backend.ai.utils.github import save_github_md_files_locally
from backend.ai.utils.text import chunk_file_contents
from backend.config import settings

from superduperdb.container.document import Document
from superduperdb.db.mongodb.query import Collection


def _create_ai_text_artifacts(repo):
    files = save_github_md_files_locally(repo)
    # Chunked text is more suitable input for the AI models
    ai_text_artifacts = chunk_file_contents(repo, files)
    return ai_text_artifacts


def load_ai_artifacts(db):
    db_artifacts = db.show('vector_index')
    for repo in settings.default_repos:
        if repo in db_artifacts:
            continue

        artifacts = _create_ai_text_artifacts(repo)
        documents = [
            Document(
                {settings.vector_embedding_key: row['text'], 'src_url': row['src_url']}
            )
            for _, row in artifacts.iterrows()
        ]
        db.execute(Collection(name=repo).insert_many(documents))
