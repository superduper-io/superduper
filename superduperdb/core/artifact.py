import typing as t
import typing_extensions as te
import uuid

from superduperdb.datalayer.base.artifacts import ArtifactStore
from superduperdb.misc.serialization import Info, serializers

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
    info: Info = None

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
        if isinstance(self.artifact, list):
            # TODO: BROKEN! we should hash the whole artifact and cache it
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
    #: A string identifying the artifact in the artifact store
    file_id: str

    #: An optional dictionary used to create the artifact
    info: Info

    #: The name of the serializer used for the artifact store
    serializer: str


# TODO: this no longer appears to be called anywhere
def load_artifact(
    desc: ArtifactDesc,
    artifact_store: ArtifactStore,
    cache: t.Dict[str, Artifact],
) -> Artifact:
    """Load an Artifact from the store

    :param desc:  Describe the artifact to be loaded
    :param artifact_store:  The store holding the artifact
    :param cache:  A cache dictionary mapping `file_id` to artifacts

    """
    file_id, info, serializer = desc['file_id'], desc['info'], desc['serializer']
    if file_id in cache:
        return cache[file_id]

    artifact = artifact_store.load_artifact(file_id, serializer, info)
    a = Artifact(artifact=artifact, info=info, serializer=serializer)

    cache[file_id] = a.artifact
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
