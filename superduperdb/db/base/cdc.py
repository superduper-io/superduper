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

    db.cdc.start(on=Collection('test_collection'))
"""

import json
import queue
import threading
import traceback
import typing as t
from abc import ABC, abstractmethod
from collections import Counter

from pymongo.change_stream import CollectionChangeStream

from superduperdb import logging
from superduperdb.container.job import FunctionJob
from superduperdb.container.task_workflow import TaskWorkflow
from superduperdb.db.base.vector_task_factory import vector_task_factory
from superduperdb.misc.runnable.queue_chunker import QueueChunker
from superduperdb.misc.runnable.runnable import Event

from .base_cdc import DBEvent, Packet

if t.TYPE_CHECKING:
    from superduperdb.db.base.db import DB
    from superduperdb.db.base.query import TableOrCollection


queue_chunker = QueueChunker(chunk_size=100, timeout=0.2)


class BaseDatabaseListener(ABC):
    """
    A Base class which defines basic functions to implement.
    """

    IDENTITY_SEP: str = '/'
    _scheduler: t.Optional[threading.Thread]

    def __init__(
        self,
        db: 'DB',
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

    def event_handler(self, change: t.Dict, event: DBEvent) -> None:
        """event_handler.
        A helper fxn to handle incoming changes from change stream on a collection.

        :param change: The change (document) observed during the change stream.
        """
        if event == DBEvent.insert:
            self._change_counters['inserts'] += 1
            self.on_create(change, db=self.db, collection=self._on_component)

        elif event == DBEvent.update:
            self._change_counters['updates'] += 1
            self.on_update(change, db=self.db, collection=self._on_component)

        elif event == DBEvent.delete:
            self._change_counters['deletes'] += 1
            self.on_delete(change, db=self.db, collection=self._on_component)


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
            cdc_stream = self.listener.setup_cdc()
            self.start_event.set()
            while not self.stop_event.is_set():
                self.listener.next_cdc(cdc_stream)
        except Exception:
            logging.error('In DatabaseListenerThreadScheduler')
            traceback.print_exc()


class CDCHandler(threading.Thread):
    """
    This class is responsible for handling the change by executing the taskflow.
    This class also extends the task graph by adding funcation job node which
    does post model executiong jobs, i.e `copy_vectors`.
    """

    def __init__(self, db: 'DB', stop_event: Event, queue):
        """__init__.

        :param db: a superduperdb instance.
        :param stop_event: A threading event flag to notify for stoppage.
        """
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
                _submit_task_workflow(self.db, Packet.collate(c))

        except Exception as exc:
            traceback.print_exc()
            logging.info(f'Error while handling cdc batches :: reason {exc}')
        finally:
            self._is_running = False


def _submit_task_workflow(db: 'DB', packet: Packet) -> None:
    """
    Build a taskflow and execute it with changed ids.

    This also extends the task workflow graph with a node.
    This node is responsible for applying a vector indexing listener,
    and copying the vectors into a vector search database.
    """
    if packet.is_delete:
        workflow = TaskWorkflow(db)
    else:
        workflow = db.build_task_workflow(packet.query, ids=packet.ids, verbose=False)

    task = 'delete' if packet.is_delete else 'copy'
    task_callable, task_name = vector_task_factory(task=task)

    serialized_query = packet.query.serialize() if packet.query else None

    def add_node(identifier):
        from superduperdb.container.vector_index import VectorIndex

        vi = db.load(identifier=identifier, type_id='vector_index')
        assert isinstance(vi, VectorIndex)

        assert not isinstance(vi.indexing_listener, str)
        listener_id = vi.indexing_listener.identifier

        args = [listener_id, serialized_query, packet.ids]
        job = FunctionJob(callable=task_callable, args=args)
        workflow.add_node(f'{task_name}({listener_id})', job=job)

        return listener_id

    listener_ids = [add_node(i) for i in db.show('vector_index')]
    if not packet.is_delete:
        assert listener_ids
        listener_id = listener_ids[-1]
        model, _, key = listener_id.rpartition('/')
        workflow.add_edge(f'{model}.predict({key})', f'{task_name}({listener_id})')

    workflow.run_jobs()


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
            from superduperdb.db.mongodb.cdc.db_listener import MongoDatabaseListener

            listener = MongoDatabaseListener(*args, **kwargs)
            return t.cast(DBListenerType, listener)
        else:
            raise NotImplementedError


class DatabaseChangeDataCapture:
    def __init__(self, db: 'DB'):
        self.db = db
        self._cdc_stop_event = Event()
        self.CDC_QUEUE: queue.Queue = queue.Queue()
        self.cdc_change_handler: t.Optional[CDCHandler] = None
        self._CDC_LISTENERS: t.Dict[str, BaseDatabaseListener] = {}
        self._running = False
        self._cdc_existing_collections: t.MutableSequence['TableOrCollection'] = []

    @property
    def running(self):
        return self._running

    def start(self):
        self._running = True

        # listen to existing collection without cdc enabled
        list(map(lambda on: self.listen(on), self._cdc_existing_collections))

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
        from superduperdb.db.base import backends

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
        self._CDC_LISTENERS[on.identifier] = listener

        listener.listen()
        return listener

    def stop(self, on: str):
        try:
            listener = self._CDC_LISTENERS[on]
        except KeyError:
            raise KeyError(
                f'CDC service is not yet triggered for {on} collection/table'
            )
        listener.stop()
        self._cdc_stop_event.clear()

    def stop_handler(self):
        self._cdc_stop_event.clear()
        if self.cdc_change_handler:
            self.cdc_change_handler.join()

    def add(self, collection: 'TableOrCollection'):
        if self.running:
            self.listen(collection)
        else:
            self._cdc_existing_collections.append(collection)
