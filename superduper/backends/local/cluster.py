import click

from superduper import logging
from superduper.backends.base.cluster import Cluster
from superduper.backends.local.cdc import LocalCDCBackend
from superduper.backends.local.compute import LocalComputeBackend
from superduper.backends.local.crontab import LocalCrontabBackend
from superduper.backends.local.scheduler import LocalScheduler
from superduper.backends.local.vector_search import LocalVectorSearchBackend
from superduper.base.exceptions import InvalidArguments
from superduper.misc.importing import load_plugin


class LocalCluster(Cluster):
    """Local cluster for running infra locally.
    """

    @classmethod
    def build(cls, CFG, **kwargs):
        """Build the local cluster."""
        searcher_impl = load_plugin(CFG.vector_search_engine).VectorSearcher

        # the build function must carry a DB object.
        # FIXME: make this argument explicit
        db = kwargs.get('db', None)
        if db is None:
            raise InvalidArguments("The 'db' parameter is required in kwargs")

        cluster = LocalCluster(
            scheduler=LocalScheduler(db=db),
            compute=LocalComputeBackend(db=db),
            vector_search=LocalVectorSearchBackend(db=db, searcher_impl=searcher_impl),
            cdc=LocalCDCBackend(db=db),
            crontab=LocalCrontabBackend(db=db),
        )

        return cluster

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
