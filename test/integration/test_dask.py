import io
import time
import uuid
from contextlib import redirect_stdout

import pytest

from superduperdb import logging

try:
    import torch
except ImportError:
    torch = None
from contextlib import contextmanager

from superduperdb.backends.mongodb.query import Collection
from superduperdb.components.listener import Listener
from superduperdb.jobs.job import FunctionJob
from superduperdb.jobs.task_workflow import TaskWorkflow


@contextmanager
def add_and_cleanup_listener(database, collection_name):
    """Add listener to the database and remove it after the test"""
    listener_x = Listener(
        key='x',
        model='model_linear_a',
        select=Collection(identifier=collection_name).find(),
    )

    database.add(listener_x)
    yield database


@pytest.fixture
def distributed_db(database_with_default_encoders_and_model, dask_client):
    local_compute = database_with_default_encoders_and_model.get_compute()
    database_with_default_encoders_and_model.set_compute(dask_client)
    yield database_with_default_encoders_and_model
    database_with_default_encoders_and_model.set_compute(local_compute)


@pytest.mark.order(2)
@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_taskgraph_futures_with_dask(dask_client, distributed_db, fake_updates):
    collection_name = str(uuid.uuid4())
    _, graph = distributed_db.execute(
        Collection(identifier=collection_name).insert_many(fake_updates)
    )

    next(
        distributed_db.execute(
            Collection(identifier=collection_name).find({'update': True})
        )
    )
    dask_client.wait_all_pending_tasks()

    nodes = graph.G.nodes
    jobs = [nodes[node]['job'] for node in nodes]

    assert all([job.future.status == 'finished' for job in jobs])


@pytest.mark.order(1)
@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize(
    'dask_client, test_db',
    [('test_insert_with_distributed', 'test_insert_with_distributed')],
    indirect=True,
)
def test_insert_with_dask(distributed_db, dask_client, fake_updates):
    collection_name = str(uuid.uuid4())

    with add_and_cleanup_listener(
        distributed_db,
        collection_name,
    ) as db:
        # Submit job
        db.execute(Collection(identifier=collection_name).insert_many(fake_updates))

        # Barrier
        dask_client.wait_all_pending_tasks()

        # Get distributed logs
        logs = dask_client.client.get_worker_logs()

        logging.info("worker logs", logs)

        # Assert result
        q = Collection(identifier=collection_name).find({'update': True})
        r = next(db.execute(q))
        assert 'model_linear_a' in r['_outputs']['x']


@pytest.mark.order(3)
@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_dependencies_with_dask(dask_client, distributed_db):
    def test_node_1(*args, **kwargs):
        return 1

    def test_node_2(*args, **kwargs):
        return 2

    # Set Dask as Compute engine.
    # ------------------------------
    database = distributed_db

    # Build Task Graph
    # ------------------------------
    g = TaskWorkflow(database)
    g.add_node(
        'test_node_1',
        job=FunctionJob(callable=test_node_1, kwargs={}, args=[]),
    )

    g.add_node(
        'test_node_2',
        job=FunctionJob(callable=test_node_2, kwargs={}, args=[]),
    )
    g.add_edge(
        'test_node_1',
        'test_node_2',
    )

    # Run Job
    # ------------------------------
    g.run_jobs()
    dask_client.wait_all_pending_tasks()

    # Validate Output
    # ------------------------------
    futures = list(dask_client.list_all_pending_tasks().values())
    assert len(futures) == 2
    assert futures[0].status == 'finished'
    assert futures[1].status == 'finished'
    assert futures[0].result() == 1
    assert futures[1].result() == 2


@pytest.mark.order(4)
def test_model_job_logs(distributed_db, fake_updates):
    # Set Collection Listener
    # ------------------------------
    collection = Collection(identifier=str(uuid.uuid4()))

    listener_x = Listener(
        key='x',
        model='model_linear_a',
        select=collection.find(),
    )
    jobs, _ = distributed_db.add(listener_x)

    # Insert data to the Collection
    # ------------------------------
    distributed_db.execute(collection.insert_many(fake_updates))

    # Validate Log Output
    # ------------------------------
    f = io.StringIO()
    with redirect_stdout(f):
        jobs[0].watch()
    s = f.getvalue()
    logs = s.split('\n')
    retry_left = 5
    while not jobs[0].future.done() or retry_left != 0:
        time.sleep(1)
        retry_left -= 1
    assert len(logs) > 1
