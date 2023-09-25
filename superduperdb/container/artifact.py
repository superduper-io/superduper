import typing as t

import typing_extensions as te

from superduperdb.db.base.artifact import ArtifactStore
from superduperdb.misc.serialization import Info, serializers

ArtifactCache = t.Dict[int, t.Any]


class Artifact:
    """
    An artifact from a computation that can be serialized or deserialized

    :param artifact: the computed artifact, which may be of any type
    :param file_id: a key used to identify the file in the ``ArtifactStore``
    :param info: the ``info`` dictionary is passed into
                 ``ArtifactStore.create_artifact()``
    :param object_id: the Python ``id()`` of the artifact
    :param save_method: the name of the finalization method on the artifact
                        to call before saving
    :param serializer: the name of the serializer
    :param sha1: the sha1 hash of the artifact
    :param hash: in case the object isn't hashable (deduplication not possible)
    """

    artifact: t.Any = None
    file_id: t.Optional[str] = None
    info: Info = None
    object_id: int = 0
    save_method: t.Optional[str] = None
    serializer: str = 'dill'
    sha1: str = ''
    hash: t.Optional[int] = None

    def __init__(
        self,
        artifact: t.Any = None,
        file_id: t.Any = None,
        info: t.Optional[t.Dict] = None,
        object_id: int = 0,
        serializer: str = 'dill',
        sha1: str = '',
        hash: t.Optional[int] = None,
    ):
        self.artifact = artifact
        self.file_id = file_id
        self.info = info
        self.object_id = object_id
        self.serializer = serializer
        self.sha1 = sha1
        self.hash = hash

    def __hash__(self):
        if self.hash is not None:
            return self.hash
        if isinstance(self.artifact, list):
            return hash(str(self.artifact[:100]))
        if isinstance(self.artifact, dict):
            return hash(str(self.artifact))
        else:
            return hash(self.artifact)

    def __eq__(self, other):
        return self.artifact == other.artifact

    def __repr__(self):
        return f'<Artifact artifact={str(self.artifact)} serializer={self.serializer}>'

    def serialize(self) -> bytes:
        """Serialize this artifact into bytes"""
        serializer = serializers[self.serializer]
        return serializer.encode(self.artifact, self.info)

    def save(self, artifact_store: ArtifactStore) -> t.Dict[str, t.Any]:
        """Store this artifact, and return a dictionary of the results

        :param artifact_store: the store to save the Artifact in
        """
        b = self.serialize()
        file_id, sha1 = artifact_store.create_artifact(bytes=b)
        return {'file_id': file_id, 'sha1': sha1, 'serializer': self.serializer}


class ArtifactDesc(te.TypedDict):
    """A description of an artifact in an artifact store

    :param file_id: A string identifying the artifact in the artifact store
    :param info: An optional dictionary used to create the artifact
    :param serializer: The name of the serializer used for the artifact store
    """

    file_id: str
    info: Info
    serializer: str


class ArtifactSavingError(Exception):
    pass
