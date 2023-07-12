import os
from unittest.mock import patch

import pytest

from superduperdb.datalayer.mongodb.query import Collection
from superduperdb.cluster.dask.dask_client import dask_client


@pytest.fixture(scope="module")
def local_dask_client():
    from superduperdb import CFG

    for component in ['DATA_BACKEND', 'ARTIFACT', 'METADATA']:
        os.environ[f'SUPERDUPERDB_DATA_LAYERS_{component}_KWARGS_PORT'] = '27018'
        os.environ[f'SUPERDUPERDB_DATA_LAYERS_{component}_KWARGS_HOST'] = 'localhost'
        os.environ[
            f'SUPERDUPERDB_DATA_LAYERS_{component}_KWARGS_USERNAME'
        ] = 'testmongodbuser'
        os.environ[
            f'SUPERDUPERDB_DATA_LAYERS_{component}_KWARGS_PASSWORD'
        ] = 'testmongodbpassword'

        os.environ[f'SUPERDUPERDB_DATA_LAYERS_{component}_NAME'] = (
            '_filesystem:test_db' if component == "ARTIFACT" else 'test_db'
        )
    client = dask_client(CFG.dask, local=True)
    yield client
    client.shutdown()


def test_taskgraph_futures_with_dask(
    local_dask_client, a_watcher, random_data, an_update
):
    from superduperdb import CFG

    with patch.object(CFG, "distributed", True):
        random_data.distributed = True
        random_data._distributed_client = local_dask_client
        _, graph = random_data.execute(
            Collection(name='documents').insert_many(an_update)
        )

    next(random_data.execute(Collection(name='documents').find({'update': True})))
    local_dask_client.wait_all_pending_tasks()
    nodes = graph.G.nodes
    jobs = [nodes[node]['job'] for node in nodes]
    assert all([job.future.status == 'finished' for job in jobs])


def test_insert_with_dask(a_watcher, random_data, local_dask_client, an_update):
    from superduperdb import CFG

    with patch.object(CFG, "distributed", True):
        random_data.distributed = True
        random_data._distributed_client = local_dask_client
        random_data.execute(Collection(name='documents').insert_many(an_update))
        local_dask_client.wait_all_pending_tasks()
        r = next(
            random_data.execute(Collection(name='documents').find({'update': True}))
        )
    assert 'linear_a' in r['_outputs']['x']
