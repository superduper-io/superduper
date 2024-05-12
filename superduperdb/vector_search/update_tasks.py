import typing as t

from superduperdb import Document
from superduperdb.backends.base.query import Query
from superduperdb.backends.mongodb.data_backend import MongoDataBackend
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
    :param ids: List of ids which were observed as deleted documents.
    :param db: A ``DB`` instance.
    """
    return db.fast_vector_searchers[vector_index].delete(ids)


def copy_vectors(
    vector_index: str,
    query: t.Union[t.Dict, Query],
    ids: t.Sequence[str],
    db=None,
):
    """
    A helper fxn to copy vectors of a ``VectorIndex`` component
    from the databackend to the fast_vector_search backend.

    :param vector-index: A identifier of the vector-index.
    :param query: A query which was used by `db._build_task_workflow` method
    :param ids: List of ids which were observed as added/updated documents.
    :param db: A ``DB`` instance.
    """

    vi = db.vector_indices[vector_index]
    if isinstance(query, dict):
        # ruff: noqa: E501
        query: Query = Document.decode(query).unpack()  # type: ignore[no-redef]
    assert isinstance(query, Query)
    if not ids:
        select = query
    else:
        select = query.select_using_ids(ids)
    docs = db._select(select)
    docs = [doc.unpack() for doc in docs]
    key = vi.indexing_listener.key
    if '_outputs.' in key:
        key = key.split('.')[1]
    # TODO: Refactor the below logic
    vectors = []
    if isinstance(db.databackend, MongoDataBackend):
        vectors = [
            {
                'vector': MongoStyleDict(doc)[
                    f'_outputs.{vi.indexing_listener.predict_id}'
                ],
                'id': str(doc['_id']),
            }
            for doc in docs
        ]
    elif isinstance(db.databackend, IbisDataBackend):
        docs = db.execute(select.outputs(vi.indexing_listener.predict_id))
        from superduperdb.backends.ibis.data_backend import INPUT_KEY

        vectors = [
            {
                'vector': doc[f'_outputs.{vi.indexing_listener.predict_id}'],
                'id': str(doc[INPUT_KEY]),
            }
            for doc in docs
        ]
    for r in vectors:
        if hasattr(r['vector'], 'numpy'):
            r['vector'] = r['vector'].numpy()

    if vectors:
        db.fast_vector_searchers[vi.identifier].add(
            [VectorItem(**vector) for vector in vectors]
        )
