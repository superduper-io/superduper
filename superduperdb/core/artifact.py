import typing as t
import uuid

from superduperdb.datalayer.base.artifacts import ArtifactStore
from superduperdb.misc.serialization import serializers

ArtifactCache = t.Dict[int, t.Any]


class Artifact:
    """
    An artifact from a computation that can be serialized or deserialized
    """

    #: The computed artifact, which may be of any type
    artifact: t.Any = None

    #: A file_id is a key used to identify the file in the ``ArtifactStore``
    file_id: t.Optional[str] = None

    #: The ``info`` dictionary is passed into ``ArtifactStore.create_artifact()``
    info: t.Optional[t.Dict] = None

    #: The Python ``id()`` of the artifact
    object_id: int = 0

    #: The name of the finalization method on the artifact to call before saving
    save_method: t.Optional[str] = None

    #: The name of the serializer
    serializer: str = 'dill'

    #: The sha1 hash of the artifact
    sha1: str = ''

    def __init__(
        self,
        artifact: t.Any = None,
        file_id: t.Any = None,
        info: t.Optional[t.Dict] = None,
        object_id: int = 0,
        serializer: str = 'dill',
        sha1: str = '',
    ):
        self.artifact = artifact
        self.file_id = file_id
        self.info = info
        self.object_id = object_id
        self.serializer = serializer
        self.sha1 = sha1

    def __hash__(self):
        try:
            return hash(self.artifact)
        except TypeError as e:
            if isinstance(self.artifact, list):
                return hash(str(self.artifact[:100]))
            elif isinstance(self.artifact, dict):
                return hash(str(self.artifact))
            else:
                raise e

    def __eq__(self, other):
        return self.artifact == other.artifact

    def __repr__(self):
        return f'<Artifact artifact={str(self.artifact)} serializer={self.serializer}>'

    def serialize(self):
        serializer = serializers[self.serializer]
        if self.info is not None:
            bytes = serializer.encode(self.artifact, self.info)
        else:
            bytes = serializer.encode(self.artifact)
        return bytes

    def save(
        self,
        artifact_store: ArtifactStore,
    ):
        bytes = self.serialize()
        file_id, sha1 = artifact_store.create_artifact(bytes=bytes)
        return {'file_id': file_id, 'sha1': sha1, 'serializer': self.serializer}

    @staticmethod
    def load(r, artifact_store: ArtifactStore, cache):
        if r['file_id'] in cache:
            return cache[r['file_id']]
        artifact = artifact_store.load_artifact(
            r['file_id'], r['serializer'], info=r['info']
        )
        a = Artifact(
            artifact=artifact,
            serializer=r['serializer'],
            info=r['info'],
        )
        cache[r['file_id']] = a.artifact
        return a


class InMemoryArtifacts:
    def __init__(self):
        self.cache = {}

    def create_artifact(self, bytes):
        file_id = str(uuid.uuid4())
        self.cache[file_id] = bytes
        return file_id, ''

    def delete_artifact(self, file_id):
        del self.cache[file_id]


class ArtifactSavingError(Exception):
    pass
