import typing as t

import click
import numpy

from superduper import logging
from superduper.backends.base.cluster import Cluster
from superduper.backends.base.vector_search import (
    BaseVectorSearcher,
    VectorItem,
    measures,
)
from superduper.backends.local.cdc import LocalCDCBackend
from superduper.backends.local.compute import LocalComputeBackend
from superduper.backends.local.crontab import LocalCrontabBackend
from superduper.backends.local.scheduler import LocalScheduler
from superduper.backends.local.vector_search import LocalVectorSearchBackend
from superduper.misc.importing import load_plugin


class LocalCluster(Cluster):
    """Local cluster for running infra locally.

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

        return LocalCluster(
            scheduler=LocalScheduler(),
            compute=LocalComputeBackend(),
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
