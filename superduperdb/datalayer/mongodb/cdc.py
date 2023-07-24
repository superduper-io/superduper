import time
import threading
import json
import queue
import datetime
from collections import Counter
from enum import Enum
import typing as t
from abc import ABC, abstractmethod
import dataclasses as dc

from pydantic import BaseModel
from bson.objectid import ObjectId as BsonObjectId
from pymongo.change_stream import CollectionChangeStream

import superduperdb as s
from superduperdb.vector_search.base import VectorCollectionConfig, VectorCollectionItem
from superduperdb.core.serializable import Serializable
from superduperdb.misc.logger import logging
from superduperdb.datalayer.mongodb import query
from superduperdb.datalayer.base.datalayer import Datalayer
from superduperdb.core.task_workflow import TaskWorkflow
from superduperdb.core.job import FunctionJob
from superduperdb.misc.task_queue import cdc_queue
from superduperdb.core.vector_index import VectorIndex

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


class BasePacket(BaseModel):
    """
    A base packet to represent message in task queue.
    """

    event_type: str = DBEvent.insert.value
    ids: t.Sequence[t.Union[ObjectId, str]]


class ChangePacket(BasePacket):
    """
    A single packet representation of message in task queue.
    """

    query: Serializable


class BatchPacket(ChangePacket):
    """
    A batch of packets to be transfered to task queue.
    """

    pass


@dc.dataclass
class MongoChangePipeline:
    """`MongoChangePipeline` is a class to represent watch pipeline
    in mongodb watch api.
    """

    matching_operations: t.Sequence[str] = dc.field(default_factory=list)

    def validate(self):
        raise NotImplementedError

    def build_matching(self) -> t.Sequence[t.Dict]:
        """A helper fxn to build a watch pipeline for mongo watch api.

        :param matching_operations: A list of operations to watch.
        """
        if bad := [op for op in self.matching_operations if not DBEvent.has_value(op)]:
            raise ValueError(f'Unknown operations: {bad}')

        return [{'$match': {'operationType': {'$in': [*self.matching_operations]}}}]


class ResumeToken:
    """
    A class to represent resume tokens for `MongoDatabaseWatcher`.
    """

    def __init__(self, token: TokenType) -> None:
        """__init__.

        :param token: a resume toke use to resume change stream in mongo
        :type token: `TokenType`
        """
        self._token = token

    @property
    def token(self) -> TokenType:
        return self._token


class CachedTokens:
    token_path = '.cdc.tokens'
    seperate = '\n'

    def __init__(self):
        self._current_tokens = []

    def append(self, token: TokenType) -> None:
        with open(CachedTokens.token_path, 'ab') as fp:
            stoken = json.dumps(token)
            stoken = stoken + self.seperate
            stoken = stoken.encode('utf-8')
            fp.write(stoken)  # type: ignore [arg-type]

    def load(self) -> t.Sequence[ResumeToken]:
        with open(CachedTokens.token_path, 'rb') as fp:
            jtokens = fp.read()
            tokens = jtokens.decode('utf-8')
            tokens = tokens.split(self.seperate)[:-1]
            tokens = list(map(lambda token: ResumeToken(json.loads(token)), tokens))
        self._current_tokens = tokens
        tokens = t.cast(t.Sequence[ResumeToken], tokens)
        return tokens  # type: ignore [return-value]


class BaseDatabaseWatcher(ABC):
    """
    A Base class which defines basic functions to implement.
    """

    @abstractmethod
    def watch(self):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError


def copy_vectors(
    indexing_watcher_identifier: str,
    cdc_query: Serializable,
    ids: t.Sequence[str],
    db=None,
):
    """
    A helper fxn to copy vectors of a `indexing_watcher` component/model of
    a `vector_index` watcher.

    This function will be added as node to the taskworkflow after every
    `indexing_watcher` in the defined watchers in db.
    """
    try:
        query = Serializable.deserialize(cdc_query)
        select = query.select_using_ids(ids)
        docs = db.select(select)
        docs = [doc.unpack() for doc in docs]
        model, key = indexing_watcher_identifier.split('/')  # type: ignore
        vectors = [
            {'vector': doc['_outputs'][key][model], 'id': str(doc['_id'])}
            for doc in docs
        ]
        dimensions = len(vectors[0]['vector'])
        config = VectorCollectionConfig(
            id=indexing_watcher_identifier, dimensions=dimensions
        )
        table = db.vector_database.get_table(config, create=True)

        vectors = [VectorCollectionItem(**vector) for vector in vectors]
        table.add(vectors, upsert=True)
    except Exception:
        logging.exception(
            f"Error in copying_vectors for vector_index: {indexing_watcher_identifier}"
        )
        raise


class CDCHandler(threading.Thread):
    """
    This class is responsible for handling the change by executing the taskflow.
    This class also extends the task graph by adding funcation job node which
    does post model executiong jobs, i.e `copy_vectors`.
    """

    _QUEUE_BATCH_SIZE: int = 100
    _QUEUE_TIMEOUT: int = 2

    def __init__(self, db: Datalayer, stop_event: threading.Event):
        """__init__.

        :param db: a superduperdb instance.
        :type db: BaseDatabase
        :param stop_event: A threading event flag to notify for stoppage.
        """
        self.db = db
        self._stop_event = stop_event

        threading.Thread.__init__(self, daemon=False)

    def submit_task_workflow(self, cdc_query: Serializable, ids: t.Sequence) -> None:
        """submit_task_workflow.
        A fxn to build a taskflow and execute it with changed ids.
        This also extends the task workflow graph with a node.
        This node is responsible for applying a vector indexing watcher,
        and copying the vectors into a vector search database.

        :param cdc_query: A query which will be used by `db._build_task_workflow` method
        to extract the desired data.
        :type cdc_query: Serializable
        :param ids: List of ids which were observed as changed document.
        :type ids: t.List
        :rtype: None
        """

        task_graph = self.db._build_task_workflow(cdc_query, ids=ids, verbose=False)
        task_graph = self.create_vector_watcher_task(
            task_graph, cdc_query=cdc_query, ids=ids
        )
        task_graph(self.db)

    def create_vector_watcher_task(
        self,
        task_graph: TaskWorkflow,
        cdc_query: Serializable,
        ids: t.Sequence[str],
    ) -> TaskWorkflow:
        """create_vector_watcher_task.
        A helper function to define a node in taskflow graph which is responsible for
        copying vectors to a vector db.

        :param task_graph: A DiGraph task flow which defines task on a di graph.
        :type task_graph: TaskWorkflow
        :param db: A superduperdb instance.
        :type db: 'BaseDatabase'
        :param cdc_query: A basic find query to get cursor on collection.
        :type cdc_query: Serializable

        :param ids: A list of ids observed during the change
        :type ids: t.List[str]
        :rtype: TaskWorkflow
        """
        for identifier in self.db.show('vector_index'):
            vector_index = self.db.load(identifier=identifier, variety='vector_index')
            vector_index = t.cast(VectorIndex, vector_index)
            indexing_watcher_identifier = vector_index.indexing_watcher.identifier
            task_graph.add_node(
                f'copy_vectors({indexing_watcher_identifier})',
                FunctionJob(
                    callable=copy_vectors,
                    args=[indexing_watcher_identifier, cdc_query.serialize(), ids],
                    kwargs={},
                ),
            )
            model, key = indexing_watcher_identifier.split('/')
            task_graph.add_edge(
                f'{model}.predict({key})',
                f'copy_vectors({indexing_watcher_identifier})',
            )
        return task_graph

    def on_create(self, packet: ChangePacket) -> None:
        ids = packet.ids
        cdc_query = packet.query
        self.submit_task_workflow(cdc_query=cdc_query, ids=ids)

    def on_update(self, packet: ChangePacket) -> None:
        # TODO: for now we treat updates as inserts.
        self.on_create(packet)

    def _handle(self, packet: ChangePacket) -> None:
        if packet.event_type == DBEvent.insert.value:
            self.on_create(packet)
        elif packet.event_type == DBEvent.update.value:
            self.on_update(packet)

    @staticmethod
    def _collate_packets(packets: t.Sequence[ChangePacket]) -> BatchPacket:
        """
        A helper function to coallate batch of packets into one
        `BatchPacket`.
        """

        ids = [packet.ids[0] for packet in packets]
        query = packets[0].query

        # TODO: cluster BatchPacket for each event.
        event_type = packets[0].event_type
        return BatchPacket(ids=ids, query=query, event_type=event_type)

    def get_batch_from_queue(self):
        """
        A method to get a batch of packets from task queue, with a timeout.
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
                logging.debug(f"Error while handling cdc batches :: reason {exc}")


class MongoEventMixin:
    """A Mixin class which defines helper fxns for `MongoDatabaseWatcher`
    It define basic events handling methods like
    `on_create`, `on_update`, etc.
    """

    DEFAULT_ID: str = '_id'
    EXCLUSION_KEYS: t.Sequence[str] = [DEFAULT_ID]

    def on_create(
        self, change: t.Dict, db: 'Datalayer', collection: query.Collection
    ) -> None:
        """on_create.
        A helper on create event handler which handles inserted document in the
        change stream.
        It basically extracts the change document and build the taskflow graph to
        execute.

        :param change: The changed document.
        :type change: t.Dict
        :param db: a superduperdb instance.
        :type db: 'BaseDatabase'
        :param collection: The collection on which change was observed.
        :type collection: query.Collection
        :rtype: None
        """
        logging.debug('Triggered `on_create` handler.')
        # new document added!
        document = change[CDCKeys.document_data_key.value]
        ids = [document[self.DEFAULT_ID]]
        cdc_query = collection.find()
        packet = ChangePacket(ids=ids, event_type=DBEvent.insert.value, query=cdc_query)
        cdc_queue.put_nowait(packet)

    def on_update(self, change: t.Dict, db: 'Datalayer'):
        """on_update.

        :param change:
        :type change: t.Dict
        :param db:
        :type db: 'BaseDatabase'
        """

        #  prepare updated document
        change[CDCKeys.update_descriptions_key.value]['updatedFields']
        change[CDCKeys.update_descriptions_key.value]['removedFields']


class CDCKeys(Enum):
    """
    A enum to represent mongo change document keys.
    """

    operation_type = 'operationType'
    document_key = 'documentKey'
    update_descriptions_key = 'updateDescription'
    update_field_key = 'updatedFields'
    document_data_key = 'fullDocument'


class _DatabaseWatcherThreadScheduler(threading.Thread):
    def __init__(
        self,
        watcher: BaseDatabaseWatcher,
        stop_event: threading.Event,
        start_event: threading.Event,
    ) -> None:
        threading.Thread.__init__(self, daemon=True)
        self.stop_event = stop_event
        self.start_event = start_event
        self.watcher = watcher

    def run(self) -> None:
        cdc_stream = self.watcher.setup_cdc()  # type: ignore
        self.start_event.set()
        logging.info(f'Database watch service started at {datetime.datetime.now()}')
        while not self.stop_event.is_set():
            self.watcher.next_cdc(cdc_stream)  # type: ignore
            time.sleep(0.01)


class MongoDatabaseWatcher(BaseDatabaseWatcher, MongoEventMixin):
    """
    It is a class which helps capture data from mongodb database and handle it
    accordingly.

    This class accepts options and db instance from user and starts a scheduler
    which could schedule a watching service to watch change stream.

    This class builds a workflow graph on each change observed.

    """

    IDENTITY_SEP: str = '/'

    def __init__(
        self,
        db: 'Datalayer',
        on: query.Collection,
        stop_event: threading.Event,
        identifier: 'str' = '',
        resume_token: t.Optional[ResumeToken] = None,
    ):
        """__init__.

        :param db: It is a superduperdb instance.
        :type db: 'BaseDatabase'
        :param on: It is used to define a Collection on which CDC would be performed.
        :type on: query.Collection
        :param stop_event: A threading event flag to notify for stoppage.
        :type identifier: 'threading.Event'
        :param identifier: A identifier to represent the watcher service.
        :type identifier: 'str'
        :param resume_token: A resume token is a token used to resume
        the change stream in mongo.
        :type resume_token: t.Optional[ResumeToken]
        """
        self.db = db
        self._on_component = on
        self._identifier = self._build_identifier([identifier, on.name])
        self.tokens = CachedTokens()
        self._change_counters = Counter(inserts=0, updates=0)

        self.resume_token = resume_token.token if resume_token else None
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

        :param change: change can be a prebuilt watch pipeline like
        'generic' or user Defined watch pipeline.
        """
        return MongoChangePipelines.get(change)

    def _get_reference_id(self, document: t.Dict) -> t.Optional[str]:
        """_get_reference_id.

        :param document:
        :type document: t.Dict
        :rtype: t.Optional[str]
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
        :type change: t.Dict
        :rtype: None
        """
        event = change[CDCKeys.operation_type.value]
        reference_id = self._get_reference_id(change)

        if not reference_id:
            logging.warning('Document change not handled due to no document key')
            return

        if event == DBEvent.insert.value:
            self._change_counters['inserts'] += 1
            self.on_create(change, db=self.db, collection=self._on_component)

        elif event == DBEvent.update.value:
            self._change_counters['updates'] += 1
            self.on_update(change, db=self.db)

    def dump_token(self, change: t.Dict) -> None:
        """dump_token.
        A helper utility to dump resume token from the changed document.

        :param change:
        :type change: t.Dict
        :rtype: None
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
        A method to setup cdc change stream from user provided
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
            logging.exception("Error while setting up cdc stream.")
            raise
        return stream_iterator

    def next_cdc(self, stream: CollectionChangeStream) -> None:
        """
        A method to get the next stream of change observed on the given `Collection`.
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
            logging.exception('Change stream is close or empty!, stopping cdc!')
            raise
        except Exception:
            logging.exception('Error occured during cdc!, stopping cdc!')
            raise

    def attach_scheduler(self, scheduler: threading.Thread) -> None:
        self._scheduler = scheduler

    def info(
        self,
    ) -> t.Dict:
        """
        A method to get info on the current state of watcher.
        """
        info = {}
        info.update(
            {
                'inserts': self._change_counters['inserts'],
                'updates': self._change_counters['updates'],
            }
        )
        print(json.dumps(info, indent=2))
        return info

    def set_resume_token(self, token: TokenType) -> None:
        """
        A method to set the resume token for the watcher.
        """
        self.resume_token = token

    def set_change_pipeline(
        self, change_pipeline: t.Optional[t.Union[str, t.Sequence[t.Dict]]]
    ) -> None:
        """
        A method to set the change pipeline for the watcher.
        """
        if change_pipeline is None:
            self._change_pipeline = MongoChangePipelines.get('generic')
        else:
            self._change_pipeline = change_pipeline

    def resume(self, token: ResumeToken) -> None:
        """
        A method to resume the watcher from a given token.
        """
        self.set_resume_token(token.token)
        self.watch()

    def watch(
        self, change_pipeline: t.Optional[t.Union[str, t.Sequence[t.Dict]]] = None
    ) -> None:
        """Primary fxn to initiate watching of a database on the collection
        with defined `change_pipeline` by the user.

        :param change_pipeline: A mongo watch pipeline defined by the user
        for more fine grained watching.
        """
        try:
            s.CFG.cdc = True
            self._stop_event.clear()
            if self._scheduler:
                if self._scheduler.is_alive():
                    raise RuntimeError(
                        'CDC Watcher thread is already running!,/'
                        'Please stop the watcher first.'
                    )

            if not self._cdc_change_handler.is_alive():
                del self._cdc_change_handler
                self.start_handler()

            scheduler = _DatabaseWatcherThreadScheduler(
                self, stop_event=self._stop_event, start_event=self._startup_event
            )
            self.attach_scheduler(scheduler)
            self.set_change_pipeline(change_pipeline)
            self._scheduler.start()

            while not self._startup_event.is_set():
                time.sleep(0.1)
        except Exception:
            logging.exception('Watching service stopped!')
            s.CFG.cdc = False
            self.stop()
            raise

    def last_resume_token(self) -> ResumeToken:
        """
        A method to get the last resume token from the change stream.
        """
        return self.tokens.load()[0]

    def resume_tokens(self) -> t.Sequence[ResumeToken]:
        """
        A method to get the resume tokens from the change stream.
        """
        return self.tokens.load()

    def stop(self) -> None:
        """
        A method to stop watching cdc changes.
        This stops the corresponding services as well.
        """
        s.CFG.cdc = False
        self._stop_event.set()
        self._cdc_change_handler.join()
        if self._scheduler:
            self._scheduler.join()

    def is_available(self):
        """
        A method to get the status of watcher.
        """
        return not self._stop_event.is_set()

    def close(self) -> None:
        self.stop()
