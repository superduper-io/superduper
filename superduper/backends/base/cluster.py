from abc import ABC, abstractmethod
import dataclasses as dc

from superduper.backends.base.cache import Cache
from superduper.backends.base.cdc import CDCBackend
from superduper.backends.base.compute import ComputeBackend
from superduper.backends.base.crontab import CrontabBackend
from superduper.backends.base.queue import BaseQueuePublisher
from superduper.backends.base.vector_search import VectorSearchBackend

from superduper import CFG


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

    def drop(self, force: bool = False):
        self.compute.drop()
        self.queue.drop()
        self.vector_search.drop()
        self.cdc.drop()
        self.crontab.drop()

    def disconnect(self):
        pass

    def __post_init__(self):
        self._db = None

    @classmethod
    @abstractmethod
    def build(cls, CFG):
        pass

    @property
    def db(self):
        return self._db

    @db.setter
    def db(self, value):
        self._db = value
        self.cache.db = value
        self.queue.db = value
        self.compute.db = value
        self.vector_search.db = value
        self.crontab.db = value
        self.cdc.db = value
        #self.vector_search.initialize()
