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
    serializer: str = 'pickle'

    #: The sha1 hash of the artifact
    sha1: str = ''

    def __init__(
        self,
        artifact: t.Any = None,
        file_id: t.Any = None,
        info: t.Optional[t.Dict] = None,
        object_id: int = 0,
        serializer: str = 'pickle',
        sha1: str = '',
    ):
        self.artifact = artifact
        self.file_id = file_id
        self.info = info
        self.object_id = object_id
        self.serializer = serializer
        self.sha1 = sha1

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
        cache: ArtifactCache,
        artifact_store: t.Optional[ArtifactStore] = None,
        replace=False,
    ) -> t.Dict[str, t.Any]:
        """Fill in this Artifact and save it in an ArtifactStore.

        :param artifact_store
            The store that will hold this Artifact

        :param cache
            An artifact cache

        :param replace
            If False, trying to replace an artifact in the store raises an Exception
        """
        self.object_id = id(self.artifact)
        try:
            return cache[self.object_id]
        except KeyError:
            pass

        file_id, sha1 = artifact_store.create_artifact(bytes=self.serialize())
        if self.file_id is not None:
            if replace:
                artifact_store.delete_artifact(self.file_id)
            else:
                raise ArtifactSavingError(f"Artifact {self.artifact} already saved.")

        self.file_id = file_id
        cache[self.object_id] = cache_contents = self.cache_contents()
        return cache_contents

    def cache_contents(self) -> t.Dict[str, t.Any]:
        # TODO: trying to stash the actual artifact in this cache, which is the
        # only thing that would make it useful, results in obscure test
        # breakages elsewhere in the code.
        #
        # It is likely that other parts of the code are using this cache in an
        # undisciplined fashion.  We should formalize this cache.
        return {
            # 'artifact': self.artifact,  :-(
            'file_id': self.file_id,
            'info': self.info,
            'object_id': self.object_id,
            'serializer': self.serializer,
            'sha1': self.sha1,
        }


def load_artifact(
    r: t.Dict[str, t.Any],
    artifact_store: ArtifactStore,
    cache: ArtifactCache,
) -> Artifact:
    """Load this artifact from an ArtifactStore.

    :param artifact_store
        The store that will hold this Artifact

    :param cache
        An artifact cache
    """
    r = dict(r)
    object_id = r.pop('object_id', None)
    try:
        # TODO: this won't be any use without 'artifact' being set
        return Artifact(**cache[object_id])
    except KeyError:
        pass

    # TODO: if `file_id` is not popped but passed to the constructor of Artifact,
    # it causes seemingly unrelated test failures elsewhere.  Why?
    file_id = r.pop('file_id')
    artifact = artifact_store.load_artifact(file_id, r['serializer'], info=r['info'])

    a = Artifact(artifact=artifact, object_id=id(artifact), **r)
    cache[object_id] = a.cache_contents()
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
