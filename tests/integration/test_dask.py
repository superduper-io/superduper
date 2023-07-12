import time
from unittest.mock import patch

import pytest

from superduperdb.datalayer.mongodb.query import Collection
from superduperdb.cluster.dask.dask_client import dask_client


def test_taskgraph_futures_with_dask(random_data, a_watcher, an_update):
    from superduperdb.misc.configs import CONFIG
    with patch.object(CONFIG.config, "remote", True):
        random_data.remote = True
        client = dask_client(CONFIG.config.dask)
        random_data._distributed_client = client
        output, graph = random_data.execute(Collection(name='documents').insert_many(an_update))

    client.wait_all_pending_task()
    time.sleep(3)
    nodes = graph.G.nodes
    jobs = [nodes[node]['job'] for node in nodes]
    assert all([job.future.status == 'finished' for job in jobs])

def test_insert_with_dask(random_data, a_watcher, an_update):
    from superduperdb.misc.configs import CONFIG
    with patch.object(CONFIG.config, "remote", True):
        random_data.remote = True
        client = dask_client(CONFIG.config.dask)
        random_data._distributed_client = client
        random_data.execute(Collection(name='documents').insert_many(an_update))
        time.sleep(3)
        r = next(random_data.execute(Collection(name='documents').find({'update': True})))
        assert 'linear_a' in r['_outputs']['x']

