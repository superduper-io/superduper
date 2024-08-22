import typing as t

from superduper import CFG, Document, logging
from superduper.backends.base.query import Query
from superduper.misc.special_dicts import MongoStyleDict
from superduper.vector_search.base import VectorItem

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


def delete_vectors(vector_index: str, ids: t.Sequence[str], db: 'Datalayer', **kwargs):
    """Delete vectors of a ``VectorIndex`` component in the fast_vector_search backend.

    :param vector_index: A identifier of vector-index.
    :param ids: List of ids which were observed as deleted documents.
    :param db: Datalayer instance.
    :param kwargs: Optinal **kwargs
    """
    return db.fast_vector_searchers[vector_index].delete(ids)


def copy_vectors(
    vector_index: str,
    query: t.Union[t.Dict, Query],
    ids: t.Sequence[str],
    db: 'Datalayer',
):
    """Copy vectors of a ``VectorIndex`` component from the databackend to the fast_vector_search backend.

    :param vector_index: A identifier of the vector-index.
    :param query: A query which was used by `db._build_task_workflow` method
    :param ids: List of ids which were observed as added/updated documents.
    :param db: Datalayer instance.
    """
    vi = db.vector_indices[vector_index]
    if isinstance(query, dict):
        # ruff: noqa: E501
        query: Query = Document.decode(query).unpack()  # type: ignore[no-redef]
    assert isinstance(query, Query)
    query.db = db

    if not ids:
        select = query
    else:
        select = query.select_using_ids(ids)

    docs = select.execute()
    docs = [doc.unpack() for doc in docs]

    vectors = []
    nokeys = 0
    for doc in docs:
        try:
            vector = MongoStyleDict(doc)[
                f'{CFG.output_prefix}{vi.indexing_listener.predict_id}'
            ]
        except KeyError:
            nokeys += 1
            continue
        vectors.append(
            {
                'vector': vector,
                'id': str(doc['_source']),
            }
        )
    if nokeys:
        logging.warn(
            f'{nokeys} outputs were missing. \n'
            'Note: This might happen in case of `VectorIndex` schedule jobs '
            'trigged before model outputs are yet to be computed.'
        )

    for r in vectors:
        if hasattr(r['vector'], 'numpy'):
            r['vector'] = r['vector'].numpy()

    if vectors:
        db.fast_vector_searchers[vi.identifier].add(
            [VectorItem(**vector) for vector in vectors]
        )
