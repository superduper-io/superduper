import hashlib
import typing as t

import typing_extensions as te

from superduperdb.misc.serialization import Info, serializers

if t.TYPE_CHECKING:
    from superduperdb.db.base.artifact import ArtifactStore

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
        self.hash = hash
        self._sha1 = sha1

    @property
    def sha1(self):
        if not self._sha1:
            b = self.serialize()
            self._sha1 = hashlib.sha1(b).hexdigest()
            return self._sha1
        return self._sha1

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

    @staticmethod
    def _is_self_serializable(object):
        if 'serialize' in dir(object) and 'deserialize' in dir(object):
            return True
        return False

    def serialize(self) -> bytes:
        """Serialize this artifact into bytes"""

        if self._is_self_serializable(self.artifact):
            assert hasattr(object, 'serialize')
            return t.cast(bytes, object.serialize())

        serializer = serializers[self.serializer]
        return serializer.encode(self.artifact, self.info)

    def save(self, artifact_store: 'ArtifactStore') -> t.Dict[str, t.Any]:
        """Store this artifact, and return a dictionary of the results

        :param artifact_store: the store to save the Artifact in
        """
        b = self.serialize()
        file_id = artifact_store.create(bytes=b)
        return {'file_id': file_id, 'sha1': self.sha1, 'serializer': self.serializer}


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
