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
    listener = DatabaseListener(db=db, on=Collection('test_collection'))
    listener.listen()
"""

import typing as t

from superduperdb.db.base import backends
from superduperdb.db.base.db import DB
from superduperdb.db.mongodb.cdc.base import BaseDatabaseListener
from superduperdb.db.mongodb.cdc.db_listener import MongoDatabaseListener
from superduperdb.db.mongodb.query import Collection
from superduperdb.misc.runnable.runnable import Event

DBListenerType = t.TypeVar('DBListenerType')


class DatabaseListenerFactory(t.Generic[DBListenerType]):
    """A Factory class to create instance of DatabaseListener corresponding to the
    `db_type`.
    """

    SUPPORTED_LISTENERS: t.List[str] = ['mongodb']

    def __init__(self, db_type: str = 'mongodb'):
        if db_type not in self.SUPPORTED_LISTENERS:
            raise ValueError(f'{db_type} is not supported yet for CDC.')
        self.listener = db_type

    def create(self, *args, **kwargs) -> DBListenerType:
        stop_event = Event()
        kwargs['stop_event'] = stop_event
        listener = MongoDatabaseListener(*args, **kwargs)
        return t.cast(DBListenerType, listener)


def DatabaseListener(
    db: DB,
    on: Collection,
    identifier: str = '',
    *args,
    **kwargs,
) -> BaseDatabaseListener:
    """
    Create an instance of ``BaseDatabaseListener``.
    Not to be confused with ``superduperdb.container.listener.Listener``.

    :param db: A superduperdb instance.
    :param on: Which collection/table listener service this be invoked on?
    :param identifier: A identity given to the listener service.
    """
    it = backends.data_backends.items()
    if types := [k for k, v in it if isinstance(db.databackend, v)]:
        db_type = types[0]
    else:
        raise ValueError('No backends found')

    if db_type != 'mongodb':
        raise NotImplementedError(f'Database {db_type} not supported yet!')

    factory_factory = DatabaseListenerFactory[MongoDatabaseListener]
    db_factory = factory_factory(db_type=db_type)
    return db_factory.create(db=db, on=on, identifier=identifier, *args, **kwargs)
