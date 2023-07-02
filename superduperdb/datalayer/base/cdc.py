import threading
import typing as T
import datetime

from superduperdb.queries.mongodb.queries import Collection
from superduperdb.misc.logger import logging as logger
from superduperdb.datalayer.mongodb.cdc import GenericDatabaseWatch
from superduperdb.datalayer.mongodb.cdc import MongoDatabaseWatcher
from superduperdb.misc.docs import api

DBWatcherType = T.TypeVar("DBWatcherType", bound=GenericDatabaseWatch)


class _DatabaseWatcherThreadScheduler(threading.Thread):
    def __init__(self, watcher) -> None:
        threading.Thread.__init__(self, daemon=True)
        self.watcher = watcher
        logger.info(f"Database Watch service started at {datetime.datetime.now()}")

    def run(self) -> None:
        self.watcher.cdc()

    def close(self) -> None:
        self.watcher.close()


class DatabaseWatcherFactory(T.Generic[DBWatcherType]):
    SUPPORTED_WATCHERS: T.List[T.Text] = ['mongodb']

    def __init__(self, db_type: T.Text = 'mongodb'):
        if db_type not in self.SUPPORTED_WATCHERS:
            raise NotImplementedError(f"{db_type} is not supported yet for CDC.")
        self.watcher = db_type

    def create(self, *args, **kwargs) -> T.Optional[DBWatcherType]:
        if self.watcher == "mongo":
            watcher = MongoDatabaseWatcher(*args, **kwargs)
            scheduler = _DatabaseWatcherThreadScheduler(watcher)
            watcher.attach_scheduler(scheduler)
            return watcher


@api("alpha")
class DatabaseWatcher:
    identity_sep = '/'

    def __new__(
        cls, db: 'BaseDatabase', on: Collection = None, *args, **kwargs
    ) -> DBWatcherType:
        assert on is not None, "`DatabaseWatcher` needs a source collection to watch."
        # TODO: check what the db variety
        db_type = db.variety
        watcher_type = GenericDatabaseWatch
        if db_type == "mongodb":
            watcher_type = MongoDatabaseWatcher
        db_factory = DatabaseWatcherFactory[watcher_type](db_type=db_type)
        return db_factory.create(db=db, on=on, *args, **kwargs)
