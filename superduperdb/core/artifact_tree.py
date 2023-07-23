import typing as t

from superduperdb.datalayer.base.artifacts import ArtifactStore
from superduperdb.misc.serialization import serializers
from superduperdb.misc.tree import tree_find, tree_rewrite
from .artifact import Artifact

"""
A collection of functions that process trees made up of dicts and lists
which contain Artifacts
"""

Info = t.Dict[int, t.Dict[str, t.Any]]


def put_artifacts_back(
    tree: t.Any,
    cache: t.Optional[t.Dict[str, Artifact]],
    artifact_store: t.Optional[ArtifactStore] = None,
):
    """
    Recursively search through a tree for ArtifactDescs by an Artifact with
    that `file_id`.

    :param tree: A tree made up of dicts and lists
    :param cache: A cache dictionary of Artifacts, keyed by file_id
    :param artifact_store: A store which permanently keeps the artifacts somewhere
    """

    def rewrite(t):
        file_id, serializer = t['file_id'], t['serializer']

        try:
            return cache[file_id]
        except KeyError:
            pass

        artifact = Artifact(
            artifact_store.load_artifact(file_id=file_id, serializer=serializer),
            serializer=serializer,
            info=t.get('info'),
        )
        cache[file_id] = artifact
        return artifact

    return tree_rewrite(tree, _has_file_id, rewrite)


def get_artifacts(tree: t.Any) -> t.Iterator[Artifact]:
    """Yield all Artifacts in a tree

    :param tree: A tree made up of dicts and lists
    """
    yield from tree_find(tree, _is_artifact)


def infer_artifacts(tree: t.Any) -> t.Iterator:
    """Yield all entries keyed with 'file_id' in a tree

    :param tree: A tree made up of dicts and lists
    """

    return (t['file_id'] for t in tree_find(tree, _has_file_id))


def replace_artifacts_with_dict(tree: t.Any, info: t.Dict[Artifact, str]) -> t.Any:
    """Replace every Artifact in a tree with an ArtifactDesc

    :param tree: A tree made up of dicts and lists
    :param info: A dictionary mapping artifacts to file_ids
    """

    def rewrite(t):
        return {'file_id': info[t], 'sha1': t.sha1, 'serializer': t.serializer}

    return tree_rewrite(tree, _is_artifact, rewrite)


def replace_artifacts(tree: t.Any, info: Info) -> None:
    """Replace every Artifact in a tree with hash of the value, looked up in Info

    Better explanation TBD

    :param tree: A tree made up of dicts and lists
    :param info: A dictionary mapping hashes to [TBD]
    """
    def rewrite(t):
        return info[hash(t)]

    return tree_rewrite(tree, _is_artifact, rewrite)


def load_artifacts(
    tree: t.Any, getter: t.Callable[[str], bytes], cache: t.Dict[str, Artifact]
) -> t.Any:
    """Replace ArtifactDesc dicts in a tree by Artifacts.

    Inverts replace_artifacts() above.

    Differs from `put_artifacts_back` TBD

    :param tree: A tree made up of dicts and lists
    :param getter: A function that returns `bytes` given a file_id
    :param cache: A dictionary caching artifacts by file_id
    """
    def rewrite(t):
        file_id, serializer = t['file_id'], t['serializer']
        try:
            result = cache[file_id]
        except KeyError:
            pass
        else:
            if result.serializer == serializer:
                return result
            raise ValueError(f'Wrong serializer: {result.serializer} != {serializer}')

        bytes_ = getter(file_id)
        artifact = serializers[serializer].decode(bytes_)
        cache[file_id] = Artifact(artifact=artifact, serializer=serializer)
        return cache[file_id]

    return tree_rewrite(tree, _has_file_id, rewrite)


def _is_artifact(t: t.Any) -> bool:
    return isinstance(t, Artifact)


def _has_file_id(t: t.Any) -> bool:
    return isinstance(t, dict) and 'file_id' in t
