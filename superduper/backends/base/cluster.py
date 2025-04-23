import dataclasses as dc
import time
from abc import ABC, abstractmethod

import click
import deprecated

from superduper import logging
from superduper.backends.base.cdc import CDCBackend
from superduper.backends.base.compute import ComputeBackend
from superduper.backends.base.crontab import CrontabBackend
from superduper.backends.base.scheduler import BaseScheduler
from superduper.backends.base.vector_search import VectorSearchBackend


@dc.dataclass(kw_only=True)
class Cluster(ABC):
    """Cluster object for managing the backend.

    :param scheduler: The scheduler backend.
    :param vector_search: The vector search backend.
    :param compute: The compute backend.
    :param cdc: The change data capture backend.
    :param crontab: The crontab backend.
    """

    scheduler: BaseScheduler
    vector_search: VectorSearchBackend
    compute: ComputeBackend
    cdc: CDCBackend
    crontab: CrontabBackend

    def drop(self, force: bool = False):
        """Drop all backends.

        :param force: Skip confirmation.
        """
        if not force and not click.confirm(
            "Are you sure you want to drop the cluster?"
        ):
            return

        self.compute.drop()
        self.scheduler.drop()
        self.vector_search.drop()
        self.cdc.drop()
        self.crontab.drop()

    def disconnect(self):
        """Disconnect from the cluster."""
        pass

    @classmethod
    @abstractmethod
    def build(cls, CFG, **kwargs):
        """Build the cluster from configuration.

        :param CFG: configuration object
        :param kwargs: additional parameters
        """
        pass

    @deprecated.deprecated(reason="Do we need this function ?")
    def load_custom_plugins(self):
        """Load user plugins."""
        from superduper import logging

        if 'Plugin' in self._db.show('Table'):
            logging.info("Found custom plugins - loading...")
            for plugin in self._db.show('Plugin'):
                logging.info(f"Loading plugin: {plugin}")
                plugin = self._db.load('Plugin', plugin)

    def initialize(self):
        """Initialize the cluster."""
        start = time.time()

        self.scheduler.initialize()
        self.compute.initialize()
        self.vector_search.initialize()
        self.crontab.initialize()
        self.cdc.initialize()

        logging.info(f"Cluster initialized in {time.time() - start:.2f} seconds.")
