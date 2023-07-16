import threading
import time
import typing as t
import datetime

from superduperdb.datalayer.mongodb.query import Collection
from superduperdb.misc.logger import logging
from superduperdb.datalayer.base.database import BaseDatabase
from superduperdb.datalayer.mongodb import cdc
from superduperdb.datalayer.base import backends


DBWatcherType = t.TypeVar('DBWatcherType')


class _DatabaseWatcherThreadScheduler(threading.Thread):
    def __init__(
        self, watcher: cdc.BaseDatabaseWatcher, stop_event: threading.Event
    ) -> None:
        threading.Thread.__init__(self, daemon=False)
        self.stop_event = stop_event
        self.watcher = watcher
        logging.info(f'Database watch service started at {datetime.datetime.now()}')

    def run(self) -> None:
        cdc_stream = self.watcher.setup_cdc()  # type: ignore
        while not self.stop_event.is_set():
            self.watcher.next_cdc(cdc_stream)  # type: ignore
            time.sleep(0.1)


class DatabaseWatcherFactory(t.Generic[DBWatcherType]):
    """DatabaseWatcherFactory.
    A Factory class to create instance of DatabaseWatcher corresponding to the
    `db_type`.
    """

    SUPPORTED_WATCHERS: t.List[str] = ['mongodb']

    def __init__(self, db_type: str = 'mongodb'):
        if db_type not in self.SUPPORTED_WATCHERS:
            raise ValueError(f'{db_type} is not supported yet for CDC.')
        self.watcher = db_type

    def create(self, *args, **kwargs) -> t.Optional[DBWatcherType]:
        stop_event = threading.Event()
        if self.watcher == 'mongodb':
            kwargs['stop_event'] = stop_event
            watcher = cdc.MongoDatabaseWatcher(*args, **kwargs)
            scheduler = _DatabaseWatcherThreadScheduler(watcher, stop_event=stop_event)
            watcher.attach_scheduler(scheduler)
            return t.cast(DBWatcherType, watcher)
        raise NotImplementedError


class DatabaseWatcher:
    """
    Change Data Capture (CDC) is a mechanism used in database systems to track
    and capture changes made to a table or collection in real-time.
    It allows applications to stay up-to-date with the latest changes in the database
    and perform various tasks, such as data synchronization, auditing,
    or data integration. The ChangeDataCapture class is designed to
    handle CDC tasksfor a specified table/collection in a database.

    Change streams allow applications to access real-time data changes
    without the complexity and risk of tailing the oplog.
    Applications can use change streams to subscribe to all data
    changes on a single collection,a database, or an entire deployment,
    and immediately react to them.
    Because change streams use the aggregation framework, applications
    can also filter for specific changes or transform the notifications at will.

    ref: https://www.mongodb.com/docs/manual/changeStreams/

    Use this module like this::
        db = any_arbitary_database.connect(...)
        db = superduper(db)
        watcher = DatabaseWatcher(db=db, on=Collection('test_collection'))
        watcher.watch()
    """

    identity_sep = '/'

    def __new__(  # type: ignore
        cls,
        db: 'BaseDatabase',
        on: Collection,
        identifier: str = '',
        *args,
        **kwargs,
    ) -> t.Optional[cdc.BaseDatabaseWatcher]:
        """__new__.
        A method which creates instance of `BaseDatabaseWatcher` corresponding to the
        `db`.

        This returns a instance of subclass of `BaseDatabaseWatcher`

        :param db: A superduperdb instance.
        :type db: 'BaseDatabase'
        :param on: It defines on which collection/table
        watcher service should be invoke.
        :type on: Collection
        :param identifier: A identity given to the watcher service.
        :type identifier: str
        :param args:
        :param kwargs:
        :rtype: BaseDatabaseWatcher
        """
        it = backends.data_backends.items()
        if types := [k for k, v in it if isinstance(db.databackend, v)]:
            db_type = types[0]
        else:
            raise ValueError('No backends found')

        if db_type != 'mongodb':
            raise NotImplementedError(f'Database {db_type} not supported yet!')

        factory_factory = DatabaseWatcherFactory[cdc.MongoDatabaseWatcher]
        db_factory = factory_factory(db_type=db_type)
        return db_factory.create(db=db, on=on, identifier=identifier, *args, **kwargs)
