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
    db.cdc.start()
    # or
    db.cdc.listen(on=Collection('test_collection'))
"""

import dataclasses as dc
import json
import queue
import threading
import traceback
import typing as t
from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum

from pymongo.change_stream import CollectionChangeStream

from superduperdb import logging
from superduperdb.misc.runnable.queue_chunker import QueueChunker
from superduperdb.misc.runnable.runnable import Event

if t.TYPE_CHECKING:
    from superduperdb.backends.base.query import TableOrCollection
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.base.serializable import Serializable
    from superduperdb.components.listener import Listener


class DBEvent(str, Enum):
    """`DBEvent` simple enum to hold mongo basic events."""

    delete = 'delete'
    insert = 'insert'
    update = 'update'


@dc.dataclass
class Packet:
    ids: t.Any
    query: t.Optional['Serializable']
    event_type: DBEvent = DBEvent.insert

    @property
    def is_delete(self) -> bool:
        return self.event_type == DBEvent.delete

    @staticmethod
    def collate(packets: t.Sequence['Packet']) -> 'Packet':
        """
        Collate a batch of packets into one
        """
        assert packets
        ids = [packet.ids[0] for packet in packets]
        query = packets[0].query

        # TODO: cluster Packet for each event.
        event_type = packets[0].event_type
        return type(packets[0])(ids=ids, query=query, event_type=event_type)


queue_chunker = QueueChunker(chunk_size=100, timeout=0.2)


class BaseDatabaseListener(ABC):
    """
    A Base class which defines basic functions to implement.
    """

    IDENTITY_SEP: str = '/'
    _scheduler: t.Optional[threading.Thread]
    Packet: Packet

    def __init__(
        self,
        db: 'Datalayer',
        on: 'TableOrCollection',
        stop_event: Event,
        identifier: 'str' = '',
        timeout: t.Optional[float] = None,
    ):
        self.db = db
        self._on_component = on
        self._change_counters = Counter(inserts=0, updates=0, deletes=0)
        self._identifier = self._build_identifier([identifier, on.identifier])
        self._stop_event = stop_event
        self._startup_event = Event()
        self._scheduler = None
        self.timeout = timeout
        self.db_type = None

        from superduperdb.backends.base import backends
        from superduperdb.backends.mongodb.cdc.base import MongoDBPacket

        if isinstance(self.db.databackend, backends.MongoDataBackend):
            self.db_type = 'mongodb'
            self.packet = lambda ids, query, event_type: MongoDBPacket(
                ids, query, event_type
            )
        else:
            raise NotImplementedError(f'{self.db.databackend} not supported yet!')

    @property
    def identity(self) -> str:
        return self._identifier

    @classmethod
    def _build_identifier(cls, identifiers) -> str:
        """_build_identifier.

        :param identifiers: list of identifiers.
        :rtype: str
        """
        return cls.IDENTITY_SEP.join(identifiers)

    def info(self) -> t.Dict:
        """
        Get info on the current state of listener.
        """
        info = {}
        info.update(
            {
                'inserts': self._change_counters['inserts'],
                'updates': self._change_counters['updates'],
                'deletes': self._change_counters['deletes'],
            }
        )
        logging.info(json.dumps(info, indent=2))
        return info

    @abstractmethod
    def listen(self):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError

    @abstractmethod
    def setup_cdc(self) -> CollectionChangeStream:
        raise NotImplementedError

    @abstractmethod
    def on_create(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def on_update(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def on_delete(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def next_cdc(self, stream: CollectionChangeStream) -> None:
        raise NotImplementedError

    def create_event(
        self,
        ids: t.Sequence,
        db: 'Datalayer',
        table_or_collection: 'TableOrCollection',
        event: DBEvent,
    ):
        """
        A helper to create packet based on the event type and put it on the cdc queue

        :param change: The changed document.
        :param db: a superduperdb instance.
        :param table_or_collection: The collection on which change was observed.
        :param event: CDC event type
        """
        cdc_query = None
        # TODO why was this logic here? Why is the query always the same?
        # if event != DBEvent.delete:
        cdc_query = table_or_collection.find()

        db.cdc.CDC_QUEUE.put_nowait(self.packet(ids, cdc_query, event))

    def event_handler(self, ids: t.Sequence, event: DBEvent) -> None:
        """event_handler.
        A helper fxn to handle incoming changes from change stream on a collection.

        :param ids: Changed document ids
        :param event: CDC event
        """
        if event == DBEvent.insert:
            self._change_counters['inserts'] += 1
            self.on_create(ids, db=self.db, collection=self._on_component)

        elif event == DBEvent.update:
            self._change_counters['updates'] += 1
            self.on_update(ids, db=self.db, collection=self._on_component)

        elif event == DBEvent.delete:
            self._change_counters['deletes'] += 1
            self.on_delete(ids, db=self.db, collection=self._on_component)


class DatabaseListenerThreadScheduler(threading.Thread):
    def __init__(
        self,
        listener: BaseDatabaseListener,
        stop_event: Event,
        start_event: Event,
    ) -> None:
        threading.Thread.__init__(self, daemon=True)
        self.stop_event = stop_event
        self.start_event = start_event
        self.listener = listener

    def run(self) -> None:
        try:
            self.listener.db.rebuild()
            cdc_stream = self.listener.setup_cdc()
            self.start_event.set()
            while not self.stop_event.is_set():
                self.listener.next_cdc(cdc_stream)
        except Exception as e:
            logging.error('In DatabaseListenerThreadScheduler', str(e))
            traceback.print_exc()


class CDCHandler(threading.Thread):
    """
    This class is responsible for handling the change by executing the taskflow.
    This class also extends the task graph by adding funcation job node which
    does post model executiong jobs, i.e `copy_vectors`.
    """

    def __init__(self, db: 'Datalayer', stop_event: Event, queue):
        """__init__.

        :param db: a superduperdb instance.
        :param stop_event: A threading event flag to notify for stoppage.
        """
        db.rebuild()

        self.db = db
        self._stop_event = stop_event
        self._is_running = False
        self.cdc_queue = queue
        threading.Thread.__init__(self, daemon=True)

    @property
    def is_running(self):
        return self._is_running

    def run(self):
        self._is_running = True
        try:
            for c in queue_chunker(self.cdc_queue, self._stop_event):
                packet = Packet.collate(c)
                if packet.is_delete:
                    self.db.refresh_after_delete(packet.query, packet.ids)
                else:
                    self.db.refresh_after_update_or_insert(
                        packet.query,
                        packet.ids,
                    )

        except Exception as exc:
            logging.error("Error while handling cdc batches :: reason", exc)
            traceback.print_exc()
        finally:
            self._is_running = False


DBListenerType = t.TypeVar('DBListenerType')


class DatabaseListenerFactory(t.Generic[DBListenerType]):
    """A Factory class to create instance of DatabaseListener corresponding to the
    `db_type`.
    """

    SUPPORTED_LISTENERS: t.List[str] = ['mongodb']

    def __init__(self, db_type: str = 'mongodb'):
        if db_type not in self.SUPPORTED_LISTENERS:
            raise ValueError(f'{db_type} is not supported yet for CDC.')
        self.db_type = db_type

    def create(self, *args, **kwargs) -> DBListenerType:
        stop_event = Event()
        kwargs['stop_event'] = stop_event
        if self.db_type == 'mongodb':
            from superduperdb.backends.mongodb.cdc.listener import MongoDatabaseListener

            listener = MongoDatabaseListener(*args, **kwargs)
            return t.cast(DBListenerType, listener)
        else:
            raise NotImplementedError


class DatabaseChangeDataCapture:
    """
    DatabaseChangeDataCapture is a Python class that provides a flexible and
    extensible framework for capturing and managing data changes
    in a database.

    This class is repsonsible for cdc service on the provided `db` instance
    This class is designed to simplify the process of tracking changes
    to database records,allowing you to monitor and respond to
    data modifications efficiently.
    """

    def __init__(self, db: 'Datalayer'):
        self.db = db
        self._cdc_stop_event = Event()
        self.CDC_QUEUE: queue.Queue = queue.Queue()
        self.cdc_change_handler: t.Optional[CDCHandler] = None
        self._CDC_LISTENERS: t.Dict[str, BaseDatabaseListener] = {}
        self._running: bool = False
        self._cdc_existing_collections: t.MutableSequence['TableOrCollection'] = []

        listeners = self.db.show('listeners')
        if listeners:
            from superduperdb.components.listener import Listener

            for listener in listeners:
                listener = db.load(identifier=listener, type_id='listener')
                assert isinstance(listener, Listener)
                self.add(listener)

    @property
    def running(self) -> bool:
        return self._running

    def start(self):
        """
        This method starts the cdc process on the database.
        """
        self._running = True

        # listen to existing collection without cdc enabled
        for collection in self._cdc_existing_collections:
            self.listen(collection)

    def listen(
        self,
        on: 'TableOrCollection',
        identifier: str = '',
        *args,
        **kwargs,
    ):
        """
        Starts cdc service on the provided collection
        Not to be confused with ``superduperdb.container.listener.Listener``.

        :param db: A superduperdb instance.
        :param on: Which collection/table listener service this be invoked on?
        :param identifier: A identity given to the listener service.
        """
        from superduperdb.backends.base import backends

        if isinstance(self.db.databackend, backends.MongoDataBackend):
            db_type = 'mongodb'
        else:
            raise NotImplementedError(f'{self.db.databackend} not supported yet!')

        if not self.cdc_change_handler:
            cdc_change_handler = CDCHandler(
                db=self.db, stop_event=self._cdc_stop_event, queue=self.CDC_QUEUE
            )
            cdc_change_handler.start()
            self.cdc_change_handler = cdc_change_handler

        db_factory: DatabaseListenerFactory = DatabaseListenerFactory(db_type=db_type)
        listener = db_factory.create(
            db=self.db, on=on, identifier=identifier, *args, **kwargs
        )
        self._CDC_LISTENERS[on.identifier] = listener  # type: ignore[index]

        listener.listen()
        return listener

    def stop(self, name: str = ''):
        """
        Stop all registered listeners
        :param name: Listener name
        """
        try:
            if name:
                try:
                    self._CDC_LISTENERS[name].stop()
                except KeyError:
                    raise KeyError(f'{name} is already down or not added yet')
                else:
                    del self._CDC_LISTENERS[name]

            for _, listener in self._CDC_LISTENERS.items():
                listener.stop()
        finally:
            self._running = False
            self._CDC_LISTENERS = {}
            self.stop_handler()

    def stop_handler(self):
        """
        Stop the cdc handler thread
        """
        self._cdc_stop_event.set()
        if self.cdc_change_handler:
            self.cdc_change_handler.join()
        self.cdc_change_handler = None

    def add(self, listener: 'Listener'):
        """
        This method registered the given collection for cdc
        """
        collection = listener.select.table_or_collection
        if self.running and collection.identifier not in self._CDC_LISTENERS:
            self.listen(collection)
        else:
            # Append to existing collection list
            self._cdc_existing_collections.append(collection)
