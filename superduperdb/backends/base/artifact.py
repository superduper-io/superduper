import typing as t
from abc import ABC, abstractmethod

from superduperdb.base.artifact import Artifact
from superduperdb.misc.serialization import Info, serializers
from superduperdb.misc.tree import tree_find, tree_rewrite

if t.TYPE_CHECKING:
    from superduperdb.components.component import Component


def _is_artifact(t: t.Any) -> bool:
    return isinstance(t, Artifact)


def _has_file_id(t: t.Any) -> bool:
    return isinstance(t, dict) and 'file_id' in t


class ArtifactStoreMixin:
    @staticmethod
    def get_artifacts(tree: t.Any) -> t.Iterator[Artifact]:
        """Yield all Artifacts in a tree

        :param tree: A tree made up of dicts and lists
        """
        yield from tree_find(tree, _is_artifact)

    @staticmethod
    def _infer_artifacts(tree: t.Any) -> t.Iterator:
        """Yield all entries keyed with 'file_id' in a tree

        :param tree: A tree made up of dicts and lists
        """

        return (t['file_id'] for t in tree_find(tree, _has_file_id))

    @staticmethod
    def replace_artifacts_with_dict(tree: t.Any, info: t.Dict[str, str]) -> t.Any:
        """Replace every Artifact in a tree with an ArtifactDesc

        :param tree: A tree made up of dicts and lists
        :param info: A dictionary mapping artifacts to file_ids
        """

        def rewrite(t):
            return {'file_id': info[t.sha1], 'sha1': t.sha1, 'serializer': t.serializer}

        return tree_rewrite(tree, _is_artifact, rewrite)

    @staticmethod
    def load_artifacts(
        tree: t.Any,
        cache: t.Dict[str, Artifact],
        getter: t.Callable[[str], bytes],
    ) -> t.Any:
        """Replace ArtifactDesc dicts in a tree by Artifacts with a getter

        :param tree: A tree made up of dicts and lists
        :param getter: A function that returns `bytes` given a file_id
        :param cache: A dictionary caching artifacts by file_id
        """

        def make(file_id, serializer):
            bytes_ = getter(file_id)
            return serializers[serializer].decode(bytes_)

        return ArtifactStoreMixin._load_artifacts(tree, make, cache)

    @staticmethod
    def _replace_artifacts(tree: t.Any, info: Info) -> t.Any:
        """Replace every Artifact in a tree with hash of the value, looked up in Info

        :param tree: A tree made up of dicts and lists
        :param info: A dictionary mapping hashes to [TBD]
        """

        def rewrite(t):
            return info[t.sha1]

        return tree_rewrite(tree, _is_artifact, rewrite)

    @staticmethod
    def _load_artifacts_from_cache(tree: t.Any, cache: t.Dict[str, Artifact]) -> t.Any:
        """Replace ArtifactDesc dicts in a tree by Artifacts."""

        def rewrite(t):
            file_id = t['file_id']
            return cache[file_id]

        return tree_rewrite(tree, _has_file_id, rewrite)

    @staticmethod
    def _load_artifacts(
        tree: t.Any,
        make: t.Callable,
        cache: t.Dict[str, Artifact],
        artifact_store: t.Optional['ArtifactStore'] = None,
    ) -> t.Any:
        """Replace ArtifactDesc dicts in a tree by Artifacts."""

        def rewrite(t):
            file_id = t['file_id']
            try:
                return cache[file_id]
            except KeyError:
                pass
            info, serializer = t.get('info'), t['serializer']
            artifact = make(file_id=file_id, serializer=serializer)
            cache[file_id] = Artifact(
                artifact=artifact,
                info=info,
                serializer=serializer,
                artifact_store=artifact_store,
                file_id=file_id,
            )
            return cache[file_id]

        return tree_rewrite(tree, _has_file_id, rewrite)


class ArtifactStore(ABC, ArtifactStoreMixin):
    """
    Abstraction for storing large artifacts separately from primary data.

    :param conn: connection to the meta-data store
    :param name: Name to identify DB using the connection
    """

    def __init__(
        self,
        conn: t.Any,
        name: t.Optional[str] = None,
    ):
        self.name = name
        self.conn = conn

    @abstractmethod
    def url(self):
        """
        Artifact store connection url
        """
        pass

    @abstractmethod
    def delete(self, file_id: str):
        """
        Delete artifact from artifact store
        :param file_id: File id uses to identify artifact in store
        """
        pass

    @abstractmethod
    def drop(self, force: bool = False):
        """
        Drop the artifact store.

        :param force: If ``True``, don't ask for confirmation
        """
        pass

    def create(self, bytes: t.Any):
        """
        Save serialized object in the artifact store.

        :param object: Object to serialize
        :param bytes: Serialized object
        """
        return self.save_artifact(bytes)

    @abstractmethod
    def save_artifact(self, serialized: bytes) -> t.Any:
        """
        Save serialized object in the artifact store.

        :param serialized: Serialized object
        """
        pass

    @abstractmethod
    def load_bytes(self, file_id: str) -> bytes:
        """
        Load bytes from artifact store.

        :param file_id: Identifier of artifact in the store
        """
        pass

    def load_artifact(self, file_id: str, serializer: str, info: Info = None) -> t.Any:
        """
        Load artifact from artifact store, and deserialize.

        :param file_id: Identifier of artifact in the store
        :param serializer: Serializer to use for deserialization
        """
        bytes = self.load_bytes(file_id)
        serializer_function = serializers[serializer]
        return serializer_function.decode(bytes, info)

    def save(self, artifacts: t.Iterator[Artifact]) -> t.Dict:
        """
        Save list of artifacts and replace the artifacts with file reference
        :param artifacts: List of ``Artifact`` instances
        """

        artifact_details = dict()
        for a in artifacts:
            file_detail = a.save(self)
            artifact_details[file_detail['sha1']] = file_detail
        return artifact_details

    def replace(self, serialized: t.Dict, artifact_info: t.Dict) -> t.Dict:
        """Replace every Artifact in the given ``object`` serialized info with
        artifact lookup.

        :param serialized: Serialized component dict
        :param artifact_info: A dictionary mapping hashes to [TBD]
        """
        return t.cast(t.Dict, self._replace_artifacts(serialized, artifact_info))

    def update(self, object: 'Component', metadata_info: t.Dict = {}):
        """
        Update artifacts with the updated object in the artifact store.
        :param object: A ``Component`` instance
        :param metadata_info: Previous object info from metadata store
        """
        serialized, artifacts = object.serialized
        artifact_details = self.save(artifacts)
        artifacts = tuple(set(self._infer_artifacts(metadata_info)))
        for oa in artifacts:
            self.delete(oa)

        return self.replace(serialized, artifact_details)

    def load(self, info, cache: t.Dict = {}, lazy: bool = False):
        """
        Recursively search through a tree for ArtifactDescs by an Artifact with
        that `file_id`.

        :param info: A tree made up of dicts and lists
        :param cache: A cache dictionary of Artifacts, keyed by file_id
        :param lazy: If True, don't load the artifacts; used so that big objects don't
                     clutter up the memory.
        """

        def no_load(*args, **kwargs):
            pass

        make = t.cast(t.Callable, self.load_artifact if not lazy else no_load)

        return self._load_artifacts(
            tree=info,
            make=make,
            cache=cache,
            artifact_store=self,
        )

    @staticmethod
    def load_from_cache(info, cache: t.Dict = {}):
        """
        Recursively search through a tree for ArtifactDescs by an Artifact with
        that `file_id` from given cache.

        :param info: A tree made up of dicts and lists
        :param cache: A cache dictionary of Artifacts, keyed by file_id
        """
        return ArtifactStoreMixin._load_artifacts_from_cache(info, cache)

    @abstractmethod
    def disconnect(self):
        """
        Disconnect the client
        """
