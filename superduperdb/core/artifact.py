import typing as t
import uuid

from superduperdb.datalayer.base.artifacts import ArtifactStore
from superduperdb.misc.serialization import serializers

ArtifactCache = t.Dict[int, t.Any]


def put_artifacts_back(d, lookup, artifact_store: t.Optional[ArtifactStore] = None):
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict) and 'file_id' in set(v.keys()):
                if v['file_id'] in lookup:
                    d[k] = lookup[v['file_id']]
                else:
                    d[k] = Artifact(
                        artifact_store.load_artifact(
                            file_id=v['file_id'],
                            serializer=v['serializer'],
                        ),
                        serializer=v['serializer'],
                        info=v.get('info'),
                    )
                    lookup[v['file_id']] = d[k]
            else:
                d[k] = put_artifacts_back(
                    v, lookup=lookup, artifact_store=artifact_store
                )
    elif isinstance(d, list):
        for i, x in enumerate(d):
            d[i] = put_artifacts_back(x, lookup=lookup, artifact_store=artifact_store)
    return d


def get_artifacts(r):
    out = []
    if isinstance(r, Artifact):
        out.extend([r])
    elif isinstance(r, dict):
        out.extend(sum([get_artifacts(v) for v in r.values()], []))
    elif isinstance(r, list):
        out.extend(sum([get_artifacts(x) for x in r], []))
    return out


def infer_artifacts(r):
    if isinstance(r, dict):
        out = []
        for k, v in r.items():
            if isinstance(v, dict) and 'file_id' in v:
                out.append(v['file_id'])
            else:
                out.extend(infer_artifacts(v))
    if isinstance(r, list):
        return sum([infer_artifacts(x) for x in r], [])
    return []


def replace_artifacts_with_dict(d, info):
    if isinstance(d, dict):
        items = d.items() if isinstance(d, dict) else enumerate(d)
        for k, v in items:
            if isinstance(v, Artifact):
                d[k] = {
                    'file_id': info[v],
                    'sha1': v.sha1,
                    'serializer': v.serializer,
                }
            elif isinstance(v, dict) or isinstance(v, list):
                d[k] = replace_artifacts_with_dict(v, info)
    return d


def replace_artifacts(d, info):
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, Artifact):
                d[k] = info[hash(v)]
            elif isinstance(v, dict):
                replace_artifacts(v, info)
            elif isinstance(v, list):
                replace_artifacts(v, info)
    if isinstance(d, list):
        for x in d:
            replace_artifacts(x, info)


def load_artifacts(d, getter, cache):
    if isinstance(d, dict) or isinstance(d, list):
        items = d.items() if isinstance(d, dict) else enumerate(d)
        for k, v in items:
            if isinstance(v, dict) and 'file_id' in v:
                if v['file_id'] not in cache:
                    bytes = getter(v['file_id'])
                    cache[v['file_id']] = Artifact(
                        artifact=serializers[v['serializer']].decode(bytes),
                        serializer=v['serializer'],
                    )
                d[k] = cache[v['file_id']]
            elif isinstance(v, dict) or isinstance(v, list):
                d[k] = load_artifacts(v, getter, cache)
    return d


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
