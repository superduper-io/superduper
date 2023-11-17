import typing as t

from superduperdb.backends.base.query import CompoundSelect
from superduperdb.base.serializable import Serializable
from superduperdb.misc.special_dicts import MongoStyleDict
from superduperdb.vector_search.base import VectorItem


def delete_vectors(
    vector_index: str,
    ids: t.Sequence[str],
    db=None,
):
    """
    A helper fxn to delete vectors of a ``VectorIndex`` component
    in the fast_vector_search backend.

    :param vector_index: A identifier of vector-index.
    :param ids: List of ids which were observed as changed documents.
    :param db: A ``DB`` instance.
    """
    return db.fast_vector_searchers[vector_index].delete(ids)


def copy_vectors(
    vector_index: str,
    query: t.Union[t.Dict, CompoundSelect],
    ids: t.Sequence[str],
    db=None,
):
    """
    A helper fxn to copy vectors of a ``VectorIndex`` component
    from the databackend to the fast_vector_search backend.

    :param vector-index: A identifier of the vector-index.
    :param query: A query which was used by `db._build_task_workflow` method
    :param ids: List of ids which were observed as changed documents.
    :param db: A ``DB`` instance.
    """
    vi = db.vector_indices[vector_index]
    if isinstance(query, dict):
        # ruff: noqa: E501
        query: CompoundSelect = Serializable.deserialize(query)  # type: ignore[no-redef]
    assert isinstance(query, CompoundSelect)
    select = query.select_using_ids(ids)
    docs = db.select(select)
    docs = [doc.unpack() for doc in docs]
    key = vi.indexing_listener.key
    model = vi.indexing_listener.model.identifier
    version = vi.indexing_listener.model.version
    vectors = [
        {
            'vector': MongoStyleDict(doc)[f'_outputs.{key}.{model}.{version}'],
            'id': str(doc['_id']),
        }
        for doc in docs
    ]
    for r in vectors:
        if hasattr(r['vector'], 'numpy'):
            r['vector'] = r['vector'].numpy()
    db.fast_vector_searchers[vi.identifier].add(
        [VectorItem(**vector) for vector in vectors]
    )
