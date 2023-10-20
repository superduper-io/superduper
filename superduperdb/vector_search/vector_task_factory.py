import typing as t

from superduperdb.container.serializable import Serializable
from superduperdb.vector_search.base import VectorCollectionConfig, VectorCollectionItem


def delete_vectors(
    indexing_listener_identifier: str,
    cdc_query: t.Optional[Serializable],
    ids: t.Sequence[str],
    db=None,
):
    """
    A helper fxn to copy vectors of a `indexing_listener` component/model of
    a `vector_index` listener.

    This function will be added as node to the taskworkflow after every
    `indexing_listener` in the defined listeners in db.

    :param indexing_listener_identifier: A identifier of indexing listener.
    :param cdc_query: A query which will be used by `db._build_task_workflow` method
    :param ids: List of ids which were observed as changed documents.
    :param db: A ``DB`` instance.
    """
    config = VectorCollectionConfig(id=indexing_listener_identifier, dimensions=0)
    table = db.vector_database.get_table(config)
    table.delete_from_ids(ids)


def copy_vectors(
    indexing_listener_identifier: str,
    cdc_query: Serializable,
    ids: t.Sequence[str],
    db=None,
):
    """
    A helper fxn to copy vectors of a `indexing_listener` component/model of
    a `vector_index` listener.

    This function will be added as node to the taskworkflow after every
    `indexing_listener` in the defined listeners in db.

    :param indexing_listener_identifier: A identifier of indexing listener.
    :param cdc_query: A query which will be used by `db._build_task_workflow` method
    :param ids: List of ids which were observed as changed documents.
    :param db: A ``DB`` instance.
    """
    query = Serializable.deserialize(cdc_query)
    select = query.select_using_ids(ids)
    docs = db.select(select)
    docs = [doc.unpack() for doc in docs]
    model, _, key = indexing_listener_identifier.rpartition('/')
    vectors = [
        {'vector': doc['_outputs'][key][model], 'id': str(doc['_id'])} for doc in docs
    ]
    dimensions = len(vectors[0]['vector'])
    config = VectorCollectionConfig(
        id=indexing_listener_identifier, dimensions=dimensions
    )
    table = db.vector_database.get_table(config, create=True)

    vector_list = [VectorCollectionItem(**vector) for vector in vectors]
    table.add(vector_list, upsert=True)


def vector_task_factory(task: str = 'copy') -> t.Tuple[t.Callable, str]:
    if task == 'copy':
        return copy_vectors, 'copy_vectors'
    elif task == 'delete':
        return delete_vectors, 'delete_vectors'
    raise NotImplementedError(f'Unknown task: {task}')
