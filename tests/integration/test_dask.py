import uuid
from contextlib import contextmanager
from unittest.mock import patch

from superduperdb.core.task_workflow import TaskWorkflow
from superduperdb.core.job import FunctionJob
from superduperdb import CFG
from superduperdb.core.watcher import Watcher
from superduperdb.datalayer.mongodb.query import Collection


@contextmanager
def add_and_cleanup_watcher(database, collection_name):
    """Add watcher to the database and remove it after the test"""
    watcher_x = Watcher(
        key='x',
        model='model_linear_a',
        select=Collection(name=collection_name).find(),
    )

    database.add(watcher_x)
    try:
        yield database
    finally:
        database.remove('watcher', 'model_linear_a/x', force=True)


def test_taskgraph_futures_with_dask(
    local_dask_client, database_with_default_encoders_and_model, fake_updates
):
    collection_name = str(uuid.uuid4())
    with patch.object(CFG, "distributed", True):
        database_with_default_encoders_and_model.distributed = True
        database_with_default_encoders_and_model._distributed_client = local_dask_client
        _, graph = database_with_default_encoders_and_model.execute(
            Collection(name=collection_name).insert_many(fake_updates)
        )

    next(
        database_with_default_encoders_and_model.execute(
            Collection(name=collection_name).find({'update': True})
        )
    )
    local_dask_client.wait_all_pending_tasks()

    nodes = graph.G.nodes
    jobs = [nodes[node]['job'] for node in nodes]

    assert all([job.future.status == 'finished' for job in jobs])


def test_insert_with_dask(
    local_dask_client, database_with_default_encoders_and_model, fake_updates
):
    collection_name = str(uuid.uuid4())
    with patch.object(CFG, "distributed", True):
        with add_and_cleanup_watcher(
            database_with_default_encoders_and_model, collection_name
        ) as database_with_watcher:
            database_with_watcher.distributed = True
            database_with_watcher._distributed_client = local_dask_client

            database_with_watcher.execute(
                Collection(name=collection_name).insert_many(fake_updates)
            )
            local_dask_client.wait_all_pending_tasks()

            r = next(
                database_with_watcher.execute(
                    Collection(name=collection_name).find({'update': True})
                ),
            )

            assert 'model_linear_a' in r['_outputs']['x']


def test_dependencies_with_dask(
    local_dask_client, database_with_default_encoders_and_model
):
    database = database_with_default_encoders_and_model

    def test_node_1(*args, **kwargs):
        return 1

    def test_node_2(*args, **kwargs):
        return 2

    G = TaskWorkflow(database)
    G.add_node(
        'test_node_1',
        job=FunctionJob(callable=test_node_1, kwargs={}, args=[]),
    )

    G.add_node(
        'test_node_2',
        job=FunctionJob(callable=test_node_2, kwargs={}, args=[]),
    )
    G.add_edge(
        'test_node_1',
        'test_node_2',
    )
    local_dask_client.futures_collection.clear()

    database.distributed = True
    database._distributed_client = local_dask_client
    G.run_jobs(distributed=True)
    local_dask_client.wait_all_pending_tasks()
    futures = list(local_dask_client.futures_collection.values())
    assert len(futures) == 2
    assert futures[0].status == 'finished'
    assert futures[1].status == 'finished'
    assert futures[0].result() == 1
    assert futures[1].result() == 2
