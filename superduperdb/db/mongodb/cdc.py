import dataclasses as dc
import datetime
import json
import queue
import threading
import time
import traceback
import typing as t
from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum

from bson.objectid import ObjectId as BsonObjectId
from pymongo.change_stream import CollectionChangeStream

import superduperdb as s
from superduperdb import logging
from superduperdb.container.job import FunctionJob
from superduperdb.container.serializable import Serializable
from superduperdb.container.task_workflow import TaskWorkflow
from superduperdb.container.vector_index import VectorIndex
from superduperdb.db.base.db import DB
from superduperdb.db.mongodb import query
from superduperdb.misc.task_queue import cdc_queue
from superduperdb.vector_search.base import VectorCollectionConfig, VectorCollectionItem

MongoChangePipelines: t.Dict[str, t.Sequence[t.Any]] = {'generic': []}
TokenType = t.Dict[str, str]


class DBEvent(Enum):
    """`DBEvent` simple enum to hold mongo basic events."""

    delete: str = 'delete'
    insert: str = 'insert'
    update: str = 'update'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class ObjectId(BsonObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not isinstance(v, BsonObjectId):
            raise TypeError('Id is required.')
        return str(v)


@dc.dataclass
class Packet:
    """
    A base packet to represent message in task queue.
    """

    ids: t.List[t.Union[ObjectId, str]]
    query: t.Optional[Serializable]
    event_type: str = DBEvent.insert.value


@dc.dataclass
class MongoChangePipeline:
    """`MongoChangePipeline` is a class to represent listen pipeline
    in mongodb watch api.
    """

    matching_operations: t.Sequence[str] = dc.field(default_factory=list)

    def validate(self):
        raise NotImplementedError

    def build_matching(self) -> t.Sequence[t.Dict]:
        """A helper fxn to build a listen pipeline for mongo watch api.

        :param matching_operations: A list of operations to watch.
        """
        if bad := [op for op in self.matching_operations if not DBEvent.has_value(op)]:
            raise ValueError(f'Unknown operations: {bad}')

        return [{'$match': {'operationType': {'$in': [*self.matching_operations]}}}]


class CachedTokens:
    token_path = '.cdc.tokens'
    separate = '\n'

    def __init__(self):
        # BROKEN: self._current_tokens is never read from
        self._current_tokens = []

    def append(self, token: TokenType) -> None:
        with open(CachedTokens.token_path, 'a') as fp:
            stoken = json.dumps(token)
            stoken = stoken + self.separate
            fp.write(stoken)

    def load(self) -> t.Sequence[TokenType]:
        with open(CachedTokens.token_path) as fp:
            tokens = fp.read().split(self.separate)[:-1]
            self._current_tokens = [TokenType(json.loads(t)) for t in tokens]
        return self._current_tokens


class BaseDatabaseListener(ABC):
    """
    A Base class which defines basic functions to implement.
    """

    @abstractmethod
    def listen(self):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError


def delete_vectors(
    indexing_listener_identifier: str,
    cdc_query: t.Optional[Serializable],
    ids: t.Sequence[str],
    db=None,
):
    """
    A helper fxn to copy vectors of a `indexing_listener` component/model of
    a `vector_index` listener.

    This function will be added as node to the taskworkflow after every
    `indexing_listener` in the defined listeners in db.
    """
    try:
        config = VectorCollectionConfig(id=indexing_listener_identifier, dimensions=0)
        table = db.vector_database.get_table(config)
        table.delete_from_ids(ids)
    except Exception:
        logging.error(
            f"Error while deleting for vector_index: {indexing_listener_identifier}"
        )
        raise


def copy_vectors(
    indexing_listener_identifier: str,
    cdc_query: Serializable,
    ids: t.Sequence[str],
    db=None,
):
    """
    A helper fxn to copy vectors of a `indexing_listener` component/model of
    a `vector_index` listener.

    This function will be added as node to the taskworkflow after every
    `indexing_listener` in the defined listeners in db.
    """
    try:
        query = Serializable.deserialize(cdc_query)
        select = query.select_using_ids(ids)
        docs = db.select(select)
        docs = [doc.unpack() for doc in docs]
        model, key = indexing_listener_identifier.split('/')
        vectors = [
            {'vector': doc['_outputs'][key][model], 'id': str(doc['_id'])}
            for doc in docs
        ]
        dimensions = len(vectors[0]['vector'])
        config = VectorCollectionConfig(
            id=indexing_listener_identifier, dimensions=dimensions
        )
        table = db.vector_database.get_table(config, create=True)

        vector_list = [VectorCollectionItem(**vector) for vector in vectors]
        table.add(vector_list, upsert=True)
    except Exception:
        logging.error(
            f"Error in copying_vectors for vector_index: {indexing_listener_identifier}"
        )
        raise


def vector_task_factory(task: str = 'copy') -> t.Tuple[t.Callable, str]:
    if task == 'copy':
        return copy_vectors, 'copy_vectors'
    elif task == 'delete':
        return delete_vectors, 'delete_vectors'
    raise NotImplementedError(f'Unknown task: {task}')


class CDCHandler(threading.Thread):
    """
    This class is responsible for handling the change by executing the taskflow.
    This class also extends the task graph by adding funcation job node which
    does post model executiong jobs, i.e `copy_vectors`.
    """

    _QUEUE_BATCH_SIZE: int = 100
    _QUEUE_TIMEOUT: int = 2

    def __init__(self, db: DB, stop_event: threading.Event):
        """__init__.

        :param db: a superduperdb instance.
        :param stop_event: A threading event flag to notify for stoppage.
        """
        self.db = db
        self._stop_event = stop_event

        threading.Thread.__init__(self, daemon=False)

    def submit_task_workflow(
        self, cdc_query: t.Optional[Serializable], ids: t.Sequence, task: str = "copy"
    ) -> None:
        """submit_task_workflow.
        A fxn to build a taskflow and execute it with changed ids.
        This also extends the task workflow graph with a node.
        This node is responsible for applying a vector indexing listener,
        and copying the vectors into a vector search database.

        :param cdc_query: A query which will be used by `db._build_task_workflow` method
        to extract the desired data.
        :param ids: List of ids which were observed as changed document.
        :param task: A task name to be executed on vector db.
        """
        if task == "delete":
            task_workflow = TaskWorkflow(self.db)
        else:
            task_workflow = self.db._build_task_workflow(
                cdc_query, ids=ids, verbose=False
            )

        task_workflow = self.create_vector_listener_task(
            task_workflow, cdc_query=cdc_query, ids=ids, task=task
        )
        task_workflow.run_jobs()

    def create_vector_listener_task(
        self,
        task_workflow: TaskWorkflow,
        cdc_query: t.Optional[Serializable],
        ids: t.Sequence[str],
        task: str = 'copy',
    ) -> TaskWorkflow:
        """create_vector_listener_task.
        A helper function to define a node in taskflow graph which is responsible for
        executing the defined ``task`` on a vector db.

        :param task_workflow: A DiGraph task flow which defines task on a di graph.
        :param db: A superduperdb instance.
        :param cdc_query: A basic find query to get cursor on collection.
        :param ids: A list of ids observed during the change
        :param task: A task name to be executed on vector db.
        """
        task_callable, task_name = vector_task_factory(task=task)
        serialized_cdc_query = cdc_query.serialize() if cdc_query else None
        for identifier in self.db.show('vector_index'):
            vector_index = self.db.load(identifier=identifier, type_id='vector_index')
            vector_index = t.cast(VectorIndex, vector_index)
            indexing_listener_identifier = (
                vector_index.indexing_listener.identifier  # type: ignore[union-attr]
            )
            task_workflow.add_node(
                f'{task_name}({indexing_listener_identifier})',
                job=FunctionJob(
                    callable=task_callable,
                    args=[indexing_listener_identifier, serialized_cdc_query, ids],
                    kwargs={},
                ),
            )
        if task != 'delete':
            model, key = indexing_listener_identifier.split(  # type: ignore[union-attr]
                '/'
            )
            task_workflow.add_edge(
                f'{model}.predict({key})',
                f'{task_name}({indexing_listener_identifier})',
            )
        return task_workflow

    def on_create(self, packet: Packet) -> None:
        ids = packet.ids
        cdc_query = packet.query
        self.submit_task_workflow(cdc_query=cdc_query, ids=ids)

    def on_update(self, packet: Packet) -> None:
        self.on_create(packet)

    def on_delete(self, packet: Packet) -> None:
        ids = packet.ids
        cdc_query = packet.query
        self.submit_task_workflow(cdc_query=cdc_query, ids=ids, task="delete")

    def _handle(self, packet: Packet) -> None:
        if packet.event_type == DBEvent.insert.value:
            self.on_create(packet)
        elif packet.event_type == DBEvent.update.value:
            self.on_update(packet)
        elif packet.event_type == DBEvent.delete.value:
            self.on_delete(packet)

    @staticmethod
    def _collate_packets(packets: t.Sequence[Packet]) -> Packet:
        """
        A helper function to coallate batch of packets into one
        `Packet`.
        """

        ids = [packet.ids[0] for packet in packets]
        query = packets[0].query

        # TODO: cluster Packet for each event.
        event_type = packets[0].event_type
        return Packet(ids=ids, query=query, event_type=event_type)

    def get_batch_from_queue(self):
        """
        Get a batch of packets from task queue, with a timeout.
        """
        packets = []
        try:
            for _ in range(self._QUEUE_BATCH_SIZE):
                packets.append(cdc_queue.get(block=True, timeout=self._QUEUE_TIMEOUT))
                if self._stop_event.is_set():
                    return 0

        except queue.Empty:
            if len(packets) == 0:
                return None
        return CDCHandler._collate_packets(packets)

    def run(self):
        while not self._stop_event.is_set():
            try:
                packets = self.get_batch_from_queue()
                if packets:
                    self._handle(packets)
                if packets == 0:
                    break
            except Exception as exc:
                traceback.print_exc()
                logging.info(f'Error while handling cdc batches :: reason {exc}')


class MongoEventMixin:
    """A Mixin class which defines helper fxns for `MongoDatabaseListener`
    It define basic events handling methods like
    `on_create`, `on_update`, etc.
    """

    DEFAULT_ID: str = '_id'
    EXCLUSION_KEYS: t.Sequence[str] = [DEFAULT_ID]

    def on_create(self, change: t.Dict, db: DB, collection: query.Collection) -> None:
        """on_create.
        A helper on create event handler which handles inserted document in the
        change stream.
        It basically extracts the change document and build the taskflow graph to
        execute.

        :param change: The changed document.
        :param db: a superduperdb instance.
        :param collection: The collection on which change was observed.
        """
        logging.debug('Triggered `on_create` handler.')
        # new document added!
        document = change[CDCKeys.document_data_key.value]
        ids = [document[self.DEFAULT_ID]]
        cdc_query = collection.find()
        packet = Packet(ids=ids, event_type=DBEvent.insert.value, query=cdc_query)
        cdc_queue.put_nowait(packet)

    def on_update(self, change: t.Dict, db: DB, collection: query.Collection) -> None:
        """on_update.

        A helper on update event handler which handles updated document in the
        change stream.
        It basically extracts the change document and build the taskflow graph to
        execute.

        :param change: The changed document.
        :param db: a superduperdb instance.
        :param collection: The collection on which change was observed.
        """

        # TODO: Handle removed fields and updated fields.
        document = change[CDCKeys.document_key.value]
        ids = [document[self.DEFAULT_ID]]
        cdc_query = collection.find()
        packet = Packet(ids=ids, event_type=DBEvent.insert.value, query=cdc_query)
        cdc_queue.put_nowait(packet)

    def on_delete(self, change: t.Dict, db: DB, collection: query.Collection) -> None:
        """on_delete.

        A helper on delete event handler which handles deleted document in the
        change stream.
        It basically extracts the change document and build the taskflow graph to
        execute.

        :param change: The changed document.
        :param db: a superduperdb instance.
        :param collection: The collection on which change was observed.
        """
        logging.debug('Triggered `on_delete` handler.')
        # new document added!
        document = change[CDCKeys.deleted_document_data_key.value]
        ids = [document[self.DEFAULT_ID]]
        packet = Packet(ids=ids, event_type=DBEvent.delete.value, query=None)
        cdc_queue.put_nowait(packet)


class CDCKeys(Enum):
    """
    A enum to represent mongo change document keys.
    """

    operation_type = 'operationType'
    document_key = 'documentKey'
    update_descriptions_key = 'updateDescription'
    update_field_key = 'updatedFields'
    document_data_key = 'fullDocument'
    deleted_document_data_key = 'documentKey'


class _DatabaseListenerThreadScheduler(threading.Thread):
    def __init__(
        self,
        listener: BaseDatabaseListener,
        stop_event: threading.Event,
        start_event: threading.Event,
    ) -> None:
        threading.Thread.__init__(self, daemon=True)
        self.stop_event = stop_event
        self.start_event = start_event
        self.listener = listener

    def run(self) -> None:
        try:
            cdc_stream = self.listener.setup_cdc()  # type: ignore[attr-defined]
            self.start_event.set()
            logging.info(
                f'Database listen service started at {datetime.datetime.now()}'
            )
        except Exception as exc:
            logging.error(f'Error while setting up cdc stream :: reason {exc}')
            return
        while not self.stop_event.is_set():
            try:
                self.listener.next_cdc(cdc_stream)  # type: ignore[attr-defined]
            except Exception as exc:
                logging.error(f'Error while listening to cdc stream :: reason {exc}')
                break
            time.sleep(0.01)


class MongoDatabaseListener(BaseDatabaseListener, MongoEventMixin):
    """
    It is a class which helps capture data from mongodb database and handle it
    accordingly.

    This class accepts options and db instance from user and starts a scheduler
    which could schedule a listening service to listen change stream.

    This class builds a workflow graph on each change observed.

    """

    IDENTITY_SEP: str = '/'
    _scheduler: t.Optional[threading.Thread]

    def __init__(
        self,
        db: DB,
        on: query.Collection,
        stop_event: threading.Event,
        identifier: 'str' = '',
        resume_token: t.Optional[TokenType] = None,
    ):
        """__init__.

        :param db: It is a superduperdb instance.
        :param on: It is used to define a Collection on which CDC would be performed.
        :param stop_event: A threading event flag to notify for stoppage.
        :param identifier: A identifier to represent the listener service.
        :param resume_token: A resume token is a token used to resume
        the change stream in mongo.
        """
        self.db = db
        self._on_component = on
        self._identifier = self._build_identifier([identifier, on.name])
        self.tokens = CachedTokens()
        self._change_counters = Counter(inserts=0, updates=0, deletes=0)

        self.resume_token = (
            resume_token.token if resume_token else None  # type: ignore[attr-defined]
        )
        self._change_pipeline = None
        self._stop_event = stop_event
        self._startup_event = threading.Event()
        self._scheduler = None
        self.start_handler()

        super().__init__()

    @property
    def identity(self) -> str:
        return self._identifier

    def start_handler(self):
        self._cdc_change_handler = CDCHandler(db=self.db, stop_event=self._stop_event)
        self._cdc_change_handler.start()

    @classmethod
    def _build_identifier(cls, identifiers) -> str:
        """_build_identifier.

        :param identifiers: list of identifiers.
        :rtype: str
        """
        return cls.IDENTITY_SEP.join(identifiers)

    @staticmethod
    def _get_stream_pipeline(
        change: str,
    ) -> t.Optional[t.Sequence[t.Any]]:
        """_get_stream_pipeline.

        :param change: change can be a prebuilt listen pipeline like
        'generic' or user Defined listen pipeline.
        """
        return MongoChangePipelines.get(change)

    def _get_reference_id(self, document: t.Dict) -> t.Optional[str]:
        """_get_reference_id.

        :param document:
        """
        try:
            document_key = document[CDCKeys.document_key.value]
            reference_id = str(document_key['_id'])
        except KeyError:
            return None
        return reference_id

    def event_handler(self, change: t.Dict) -> None:
        """event_handler.
        A helper fxn to handle incoming changes from change stream on a collection.

        :param change: The change (document) observed during the change stream.
        """
        event = change[CDCKeys.operation_type.value]
        reference_id = self._get_reference_id(change)

        if not reference_id:
            logging.warn('Document change not handled due to no document key')
            return

        if event == DBEvent.insert.value:
            self._change_counters['inserts'] += 1
            self.on_create(change, db=self.db, collection=self._on_component)

        elif event == DBEvent.update.value:
            self._change_counters['updates'] += 1
            self.on_update(change, db=self.db, collection=self._on_component)

        elif event == DBEvent.delete.value:
            self._change_counters['deletes'] += 1
            self.on_delete(change, db=self.db, collection=self._on_component)

    def dump_token(self, change: t.Dict) -> None:
        """dump_token.
        A helper utility to dump resume token from the changed document.

        :param change:
        """
        token = change[self.DEFAULT_ID]
        self.tokens.append(token)

    def check_if_taskgraph_change(self, change: t.Dict) -> bool:
        """
        A helper method to check if the cdc change is done
        by taskgraph nodes.
        """
        if change[CDCKeys.operation_type.value] == DBEvent.update.value:
            updates = change[CDCKeys.update_descriptions_key.value]
            updated_fields = updates[CDCKeys.update_field_key.value]
            return any(
                [True for k in updated_fields.keys() if k.startswith('_outputs')]
            )
        return False

    def setup_cdc(self) -> CollectionChangeStream:
        """
        Setup cdc change stream from user provided
        """
        try:
            if isinstance(self._change_pipeline, str):
                pipeline = self._get_stream_pipeline(self._change_pipeline)

            elif isinstance(self._change_pipeline, list):
                pipeline = self._change_pipeline
                if not pipeline:
                    pipeline = None
            else:
                raise TypeError(
                    'Change pipeline can be either a string or a dictionary, '
                    f'provided {type(self._change_pipeline)}'
                )
            stream = self._on_component.change_stream(
                pipeline=pipeline, resume_after=self.resume_token
            )

            stream_iterator = stream(self.db)
            logging.info(f'Started listening database with identity {self.identity}...')
        except Exception:
            logging.exception(  # type: ignore[attr-defined]
                "Error while setting up cdc stream."
            )
            raise
        return stream_iterator

    def next_cdc(self, stream: CollectionChangeStream) -> None:
        """
        Get the next stream of change observed on the given `Collection`.
        """

        try:
            change = stream.try_next()
            if change is not None:
                logging.debug(
                    f'Database change encountered at {datetime.datetime.now()}'
                )

                if not self.check_if_taskgraph_change(change):
                    self.event_handler(change)
                self.dump_token(change)
        except StopIteration:
            logging.exception(  # type: ignore[attr-defined]
                'Change stream is close or empty!, stopping cdc!'
            )
            raise
        except Exception:
            logging.exception(  # type: ignore[attr-defined]
                'Error occured during cdc!, stopping cdc!'
            )
            raise

    def attach_scheduler(self, scheduler: threading.Thread) -> None:
        self._scheduler = scheduler

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
        print(json.dumps(info, indent=2))
        return info

    def set_resume_token(self, token: TokenType) -> None:
        """
        Set the resume token for the listener.
        """
        self.resume_token = token

    def set_change_pipeline(
        self, change_pipeline: t.Optional[t.Union[str, t.Sequence[t.Dict]]]
    ) -> None:
        """
        Set the change pipeline for the listener.
        """
        if change_pipeline is None:
            change_pipeline = MongoChangePipelines.get('generic')

        self._change_pipeline = change_pipeline  # type: ignore [assignment]

    def resume(self, token: TokenType) -> None:
        """
        Resume the listener from a given token.
        """
        self.set_resume_token(token.token)  # type: ignore[attr-defined]
        self.listen()

    def listen(
        self, change_pipeline: t.Optional[t.Union[str, t.Sequence[t.Dict]]] = None
    ) -> None:
        """Primary fxn to initiate listening of a database on the collection
        with defined `change_pipeline` by the user.

        :param change_pipeline: A mongo listen pipeline defined by the user
        for more fine grained listening.
        """
        try:
            s.CFG.cdc = True
            self._stop_event.clear()
            if self._scheduler:
                if self._scheduler.is_alive():
                    raise RuntimeError(
                        'CDC Listener thread is already running!,/'
                        'Please stop the listener first.'
                    )

            if not self._cdc_change_handler.is_alive():
                del self._cdc_change_handler
                self.start_handler()

            scheduler = _DatabaseListenerThreadScheduler(
                self, stop_event=self._stop_event, start_event=self._startup_event
            )
            self.attach_scheduler(scheduler)
            self.set_change_pipeline(change_pipeline)
            self._scheduler.start()  # type: ignore[union-attr]

            while not self._startup_event.is_set():
                time.sleep(0.1)
        except Exception:
            logging.error('Listening service stopped!')
            s.CFG.cdc = False
            self.stop()
            raise

    def last_resume_token(self) -> TokenType:
        """
        Get the last resume token from the change stream.
        """
        return self.tokens.load()[0]

    def resume_tokens(self) -> t.Sequence[TokenType]:
        """
        Get the resume tokens from the change stream.
        """
        return self.tokens.load()

    def stop(self) -> None:
        """
        Stop listening cdc changes.
        This stops the corresponding services as well.
        """
        s.CFG.cdc = False
        self._stop_event.set()
        self._cdc_change_handler.join()
        if self._scheduler:
            self._scheduler.join()

    def is_available(self) -> bool:
        """
        Get the status of listener.
        """
        return not self._stop_event.is_set()

    def close(self) -> None:
        self.stop()
