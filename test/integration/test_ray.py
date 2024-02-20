import io
import uuid
from contextlib import redirect_stdout

import pytest

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
    m = database.load('model', 'model_linear_a')
    listener_x = Listener(
        key='x',
        model=m,
        select=Collection(identifier=collection_name).find(),
    )

    database.add(listener_x)
    yield database


@pytest.fixture
def distributed_db(database_with_default_encoders_and_model, ray_client):
    local_compute = database_with_default_encoders_and_model.get_compute()
    database_with_default_encoders_and_model.set_compute(ray_client)
    yield database_with_default_encoders_and_model
    database_with_default_encoders_and_model.set_compute(local_compute)


@pytest.mark.order(2)
@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_taskgraph_futures_with_ray(ray_client, distributed_db, fake_updates):
    collection_name = str(uuid.uuid4())
    _, graph = distributed_db.execute(
        Collection(identifier=collection_name).insert_many(fake_updates)
    )

    next(
        distributed_db.execute(
            Collection(identifier=collection_name).find({'update': True})
        )
    )
    ray_client.wait_all()

    nodes = graph.G.nodes
    jobs = [nodes[node]['job'] for node in nodes]

    assert all([job.future.future().exception() is None for job in jobs])


@pytest.mark.order(1)
@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize(
    'ray_client, test_db',
    [('test_insert_with_distributed', 'test_insert_with_distributed')],
    indirect=True,
)
def test_insert_with_ray(distributed_db, ray_client, fake_updates):
    collection_name = str(uuid.uuid4())

    with add_and_cleanup_listener(
        distributed_db,
        collection_name,
    ) as db:
        # Submit job
        db.execute(Collection(identifier=collection_name).insert_many(fake_updates))

        # Barrier
        ray_client.wait_all()

        # Assert result
        q = Collection(identifier=collection_name).find({'update': True})
        r = next(db.execute(q))
        assert 'model_linear_a' in r['_outputs']['x']


@pytest.mark.order(3)
@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_dependencies_with_ray(ray_client, distributed_db):
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
    ray_client.wait_all()

    # Validate Output
    # ------------------------------
    futures = list(ray_client.tasks.values())

    assert len(futures) == 2

    assert futures[0].future().result() == 1
    assert futures[1].future().result() == 2


@pytest.mark.skip
def test_model_job_logs(distributed_db, fake_updates):
    # Set Collection Listener
    # ------------------------------

    collection = Collection(identifier=str(uuid.uuid4()))

    m = distributed_db('model', 'model_linear_a')

    listener_x = Listener(
        key='x',
        model=m,
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

    import ray

    ray.wait([job.future for job in jobs], num_returns=len(jobs), timeout=10)
    print(logs)
    assert len(logs) > 1
