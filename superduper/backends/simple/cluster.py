import typing as t

import click
import os

from superduper import logging
from superduper.backends.base.cluster import Cluster
from superduper.backends.local.cache import LocalCache
from superduper.backends.local.cdc import LocalCDCBackend
from superduper.backends.simple.compute import SimpleComputeBackend, SimpleComputeClient
from superduper.backends.local.crontab import LocalCrontabBackend
from superduper.backends.simple.scheduler import SimpleScheduler
from superduper.backends.local.vector_search import LocalVectorSearchBackend
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
        searcher_impl = load_plugin(CFG.vector_search_engine).VectorSearcher
        cache = None
        if CFG.cache and CFG.cache.startswith('redis'):
            cache = load_plugin('redis').Cache(uri=CFG.cache)
        elif CFG.cache:
            assert CFG.cache == 'in-process'
            cache = LocalCache()

        return SimpleCluster(
            cache=cache,
            scheduler=SimpleScheduler(),
            compute=SimpleComputeClient(),
            vector_search=LocalVectorSearchBackend(searcher_impl=searcher_impl),
            cdc=LocalCDCBackend(),
            crontab=LocalCrontabBackend(),
        )

    def drop(self, force: bool = False):
        """Drop the cluster.

        :param force: Force drop the cluster.
        """
        if not force:
            if not click.confirm(
                "Are you sure you want to drop the cache? ",
                default=False,
            ):
                logging.warn("Aborting...")
        if self.cache is not None:
            return self.cache.drop()