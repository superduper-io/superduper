import typing as t

from superduperdb.datalayer.base.artifacts import ArtifactStore
from superduperdb.misc.serialization import serializers
from .artifact import Artifact

"""
A collection of functions that process trees made up of dicts and lists
which contain Artifacts
"""

Info = t.Dict[int, t.Dict[str, t.Any]]


def put_artifacts_back(
    tree: t.Any,
    lookup: t.Dict[str, Artifact],
    artifact_store: t.Optional[ArtifactStore] = None,
):
    """
    Recursively search through a tree for ArtifactDescs by an Artifact with
    that `file_id`.

    :param tree: A tree made up of dicts and lists, and leaves of any type
    :param lookup: A cache dictionary of Artifacts, keyed by file_id
    :param artifact_store: A store which permanently keeps the artifacts somewhere
    """

    def to_artifact(v):
        try:
            file_id = v['file_id']
        except (KeyError, TypeError):
            return put_back(v)

        try:
            return lookup[file_id]
        except KeyError:
            pass

        artifact = Artifact(
            artifact_store.load_artifact(
                file_id=v['file_id'],
                serializer=v['serializer'],
            ),
            serializer=v['serializer'],
            info=v.get('info'),
        )
        lookup[file_id] = artifact
        return artifact

    def put_back(d):
        if isinstance(d, list):
            return [put_back(i) for i in d]
        if isinstance(d, dict):
            return {k: to_artifact(v) for k, v in d.items()}
        return d

    return put_back(tree)


def get_artifacts(tree: t.Any) -> t.Iterator['Artifact']:
    """Yield all Artifacts in a tree

    :param tree: A tree made up of dicts and lists, and leaves of any type
    """
    if isinstance(tree, Artifact):
        yield tree
    elif isinstance(tree, dict):
        yield from (a for i in tree.values() for a in get_artifacts(i))
    elif isinstance(tree, list):
        yield from (a for i in tree for a in get_artifacts(i))


def infer_artifacts(tree: t.Any) -> t.Iterator:
    """Yield all entries keyed with 'file_id' in a tree

    :param tree: A tree made up of dicts and lists, and leaves of any type
    """
    if isinstance(tree, dict) and 'file_id' in tree:
        yield tree['file_id']
    elif isinstance(tree, dict):
        yield from (a for i in tree.values() for a in infer_artifacts(i))
    elif isinstance(tree, list):
        yield from (a for i in tree for a in infer_artifacts(i))


def replace_artifacts_with_dict(tree: t.Any, info: t.Dict[Artifact, str]) -> t.Any:
    """Replace every Artifact in a tree with an ArtifactDesc

    :param tree: A tree made up of dicts and lists, and leaves of any type
    :param info: A dictionary mapping artifacts to file_ids
    """
    if isinstance(tree, dict):
        items = tree.items() if isinstance(tree, dict) else enumerate(tree)
        for k, v in items:
            if isinstance(v, Artifact):
                tree[k] = {
                    'file_id': info[v],
                    'sha1': v.sha1,
                    'serializer': v.serializer,
                }
            elif isinstance(v, dict) or isinstance(v, list):
                tree[k] = replace_artifacts_with_dict(v, info)
    return tree


def replace_artifacts(tree: t.Any, info: Info) -> None:
    """Replace every Artifact in a tree with hash of the value, looked up in Info

    Better explanation TBD

    :param tree: A tree made up of dicts and lists, and leaves of any type
    :param info: A dictionary mapping hashes to [TBD]
    """
    if isinstance(tree, dict):
        for k, v in tree.items():
            if isinstance(v, Artifact):
                tree[k] = info[hash(v)]
            elif isinstance(v, dict):
                replace_artifacts(v, info)
            elif isinstance(v, list):
                replace_artifacts(v, info)
    elif isinstance(tree, list):
        for x in tree:
            replace_artifacts(x, info)


def load_artifacts(
    tree: t.Any, getter: t.Callable[[str], bytes], cache: t.Dict[str, Artifact]
) -> t.Any:
    """Replace ArtifactDesc dicts in a tree by Artifacts.

    Inverts replace_artifacts() above.

    Differs from `put_artifacts_back` TBD

    :param tree: A tree made up of dicts and lists, and leaves of any type
    :param getter: A function that returns `bytes` given a file_id
    :param cache: A dictionary caching artifacts by file_id
    """
    if isinstance(tree, dict) or isinstance(tree, list):
        items = tree.items() if isinstance(tree, dict) else enumerate(tree)
        for k, v in items:
            if isinstance(v, dict) and 'file_id' in v:
                if v['file_id'] not in cache:
                    bytes_ = getter(v['file_id'])
                    cache[v['file_id']] = Artifact(
                        artifact=serializers[v['serializer']].decode(bytes_),
                        serializer=v['serializer'],
                    )
                tree[k] = cache[v['file_id']]
            elif isinstance(v, dict) or isinstance(v, list):
                tree[k] = load_artifacts(v, getter, cache)
    return tree
