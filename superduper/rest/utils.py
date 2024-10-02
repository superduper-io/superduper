import inspect

from superduper import Document
from superduper.components.datatype import Artifact, Encodable


def rewrite_artifacts(r, db):
    """Helper function to rewrite artifacts."""
    if isinstance(r, Encodable):
        kwargs = r.dict()
        kwargs['datatype'].encodable = 'artifact'
        blob = r._encode()[0]
        db.artifact_store.put_bytes(blob, file_id=r.identifier)
        init_args = inspect.signature(Artifact.__init__).parameters.keys()
        kwargs = {k: v for k, v in kwargs.items() if k in init_args}
        return Artifact(**kwargs)
    if isinstance(r, Document):
        return Document(rewrite_artifacts(dict(r), db=db))
    if isinstance(r, dict):
        return {k: rewrite_artifacts(v, db=db) for k, v in r.items()}
    if isinstance(r, list):
        return [rewrite_artifacts(v, db=db) for v in r]
    return r
