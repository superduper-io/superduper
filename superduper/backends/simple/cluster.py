import typing as t

import click

from superduper import logging, superduper
from superduper.backends.base.cluster import Cluster
from superduper.backends.local.cdc import LocalCDCBackend
from superduper.backends.local.crontab import LocalCrontabBackend
from superduper.backends.local.vector_search import LocalVectorSearchBackend
from superduper.backends.simple.compute import SimpleComputeBackend, SimpleComputeClient
from superduper.backends.simple.scheduler import SimpleScheduler
from superduper.backends.simple.vector_search import (
    SimpleVectorSearch,
    SimpleVectorSearchClient,
)
from superduper.misc.importing import load_plugin


class SimpleCluster(Cluster):
    """Simple cluster for running infra on a single machine.

    :param compute: The compute backend.
    :param cache: The cache backend.
    :param scheduler: The scheduler backend.
    :param vector_search: The vector search backend.
    :param cdc: The change data capture backend.
    :param crontab: The crontab backend.
    """

    @classmethod
    def build(cls, CFG, **kwargs):
        """Build the local cluster."""
        return SimpleCluster(
            scheduler=SimpleScheduler(),
            compute=SimpleComputeClient(),
            vector_search=SimpleVectorSearchClient(),
            cdc=LocalCDCBackend(),
            crontab=LocalCrontabBackend(),
        )

    def drop(self, force: bool = False):
        """Drop the cluster.

        :param force: Force drop the cluster.
        """
        if not force:
            if not click.confirm(
                "Are you sure you want to drop the cluster? ",
                default=False,
            ):
                logging.warn("Aborting...")

            self.vector_search.drop(force=True)
            self.compute.drop(force=True)


class SimpleClusterBackend(Cluster):
    """Simple cluster for running infra on a single machine.

    :param compute: The compute backend.
    :param cache: The cache backend.
    :param scheduler: The scheduler backend.
    :param vector_search: The vector search backend.
    :param cdc: The change data capture backend
    :param crontab: The crontab backend.
    """

    @classmethod
    def build(cls, CFG, **kwargs):
        """Build the local cluster."""
        searcher_impl = load_plugin(CFG.vector_search_engine).VectorSearcher

        return SimpleClusterBackend(
            scheduler=SimpleScheduler(),
            compute=SimpleComputeBackend(),
            vector_search=SimpleVectorSearch(
                backend=LocalVectorSearchBackend(searcher_impl=searcher_impl)
            ),
            cdc=LocalCDCBackend(),
            crontab=LocalCrontabBackend(),
        )

    def drop(self, force: bool = False):
        """Drop the cluster.

        :param force: Force drop the cluster.
        """
        if not force:
            if not click.confirm(
                "Are you sure you want to drop the cluster? ",
                default=False,
            ):
                logging.warn("Aborting...")

            self.vector_search.drop()
            self.compute.drop()


if __name__ == "__main__":

    import uvicorn
    from fastapi import FastAPI

    from superduper import CFG

    db = superduper(cluster_engine='local')

    app = FastAPI()

    cluster = SimpleClusterBackend.build(CFG=CFG)

    cluster.compute.build(app=app)
    cluster.vector_search.build(app=app)

    db.cluster = cluster

    cluster.db = db

    uvicorn.run(app, host='localhost', port=8001)
