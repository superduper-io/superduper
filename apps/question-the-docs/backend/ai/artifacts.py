from backend.ai.utils.github import save_github_md_files_locally
from backend.ai.utils.text import chunk_file_contents
from backend.config import settings

from superduperdb.container.document import Document
from superduperdb.db.mongodb.query import Collection


def load_ai_artifacts(db):
    db_artifacts = db.show('vector_index')
    for src in settings.documentation_sources:
        if src not in db_artifacts:
            query = Collection(name=src).insert_many(_docs(src))
            db.execute(query)


def _docs(src):
    files = save_github_md_files_locally(src)

    # Chunked text is more suitable input for the AI models
    artifacts = chunk_file_contents(src, files)
    rows = (row for _, row in artifacts.iterrows())

    key = settings.vector_embedding_key
    return [Document({key: r['text'], 'src_url': r['src_url']}) for r in rows]
