import dataclasses as dc
from abc import ABC, abstractmethod

from superduper.backends.base.cache import Cache
from superduper.backends.base.cdc import CDCBackend
from superduper.backends.base.compute import ComputeBackend
from superduper.backends.base.crontab import CrontabBackend
from superduper.backends.base.queue import BaseQueuePublisher
from superduper.backends.base.vector_search import VectorSearchBackend


@dc.dataclass(kw_only=True)
class Cluster(ABC):
    """Cluster object for managing the backend.

    :param compute: The compute backend.
    :param cache: The cache backend.
    :param queue: The queue backend.
    :param vector_search: The vector search backend.
    :param cdc: The change data capture backend.
    :param crontab: The crontab backend.
    """

    compute: ComputeBackend
    cache: Cache
    queue: BaseQueuePublisher
    vector_search: VectorSearchBackend
    cdc: CDCBackend
    crontab: CrontabBackend

    def __post_init__(self):
        self._db = None

    # TODO use the `force` parameter.
    def drop(self, force: bool = False):
        """Drop all of the backends.

        :param force: Skip confirmation.
        """
        self.compute.drop()
        self.queue.drop()
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

    @property
    def db(self):
        """Get the ``db``."""
        return self._db

    @db.setter
    def db(self, value):
        """Set the ``db``.

        :param value: ``Datalayer`` instance.
        """
        self._db = value
        self.cache.db = value
        self.queue.db = value
        self.vector_search.db = value
        self.crontab.db = value
        self.compute.db = value
        self.cdc.db = value

    def initialize(self, with_compute: bool = False):
        """Initialize the cluster.

        :param with_compute: Boolean to init
                             compute.
        """
        assert self.db
        if with_compute:
            self.compute.initialize()

        self.cache.initialize()
        self.queue.initialize()
        self.vector_search.initialize()
        self.crontab.initialize()
        self.cdc.initialize()
