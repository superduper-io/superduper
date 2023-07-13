import io
import typing as t
from superduperdb.datalayer.base.artifacts import ArtifactStore
from superduperdb.misc.serialization import serializers


class ArtifactSavingError(Exception):
    pass


class Artifact:
    def __init__(
        self,
        _artifact: t.Optional[t.Any] = None,
        serializer: str = 'pickle',
        info: t.Optional[t.Dict] = None,
        file_id: t.Optional[t.Any] = None,
    ):
        self.serializer = serializer
        self._artifact = _artifact
        self.info = info
        self.file_id = file_id

    def __repr__(self):
        return f'<Artifact artifact={str(self._artifact)} serializer={self.serializer}>'

    def save(self, artifact_store: ArtifactStore, cache, replace=False):
        object_id = id(self._artifact)
        if object_id not in cache:
            file_id, sha1 = artifact_store.create_artifact(
                object=self._artifact, serializer=self.serializer, info=self.info
            )
            if replace and self.file_id is not None:
                artifact_store.delete_artifact(self.file_id)
            elif not replace and self.file_id is not None:
                raise ArtifactSavingError(
                    "Something has gone wrong in saving, "
                    f"Artifact {self._artifact} was already saved."
                )
            self.file_id = file_id
            details = {
                'file_id': file_id,
                'sha1': sha1,
                'id': object_id,
                'serializer': self.serializer,
                'info': self.info,
            }
            cache[object_id] = details
        return cache[id(self._artifact)]

    @staticmethod
    def load(r, artifact_store: ArtifactStore, cache):
        artifact = artifact_store.load_artifact(
            r['file_id'], r['serializer'], info=r['info']
        )
        if r['file_id'] in cache:
            return cache[r['file_id']]
        a = Artifact(
            _artifact=artifact,
            serializer=r['serializer'],
            info=r['info'],
        )
        cache[r['file_id']] = a._artifact
        return a

    @property
    def a(self):
        return self._artifact

    def serialize(self):
        if self.save_method is not None:
            f = io.BytesIO()
            getattr(self._artifact, self.save_method)(f)
        return serializers[self.serializer].encode(self._artifact)
