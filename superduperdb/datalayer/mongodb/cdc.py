import pickle
import datetime
from enum import Enum
import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel

from superduperdb.misc.logger import logging
from superduperdb.queries.mongodb import queries
from superduperdb.datalayer.base.database import BaseDatabase
from superduperdb.cluster.task_workflow import TaskWorkflow

MongoChangePipelines: t.Dict[str, t.Dict] = {'generic': {}}


class DBEvent(Enum):
    """`DBEvent` simple enum to hold mongo basic events."""

    insert = 'insert'
    update = 'update'
    delete = 'delete'


class MongoChangePipeline:
    """`MongoChangePipeline` is a class to represent watch pipeline
    in mongodb watch api.
    """

    def __init__(
        self,
    ):
        self.valid_ops: t.List[DBEvent] = [
            DBEvent.insert,
            DBEvent.update,
            DBEvent.delete,
        ]

    def validate(self):
        raise NotImplementedError

    def build_matching(self, matching_operations: t.List = []) -> t.Dict:
        """A helper fxn to build a watch pipeline for mongo watch api.

        :param matching_operations: A list of operations to watch.
        :type matching_operations: t.List
        :rtype: t.Dict
        """
        if any([op for op in matching_operations if op not in self.valid_ops]):
            raise ValueError('not valid ops')

        return {'$match': {'operationType': {'$in': [*matching_operations]}}}


class ResumeToken:
    """
    A class to represent resume tokens for `MongoDatabaseWatcher`.
    """

    def __init__(self, token: str = ''):
        """__init__.

        :param token: a resume toke use to resume change stream in mongo
        :type token: str
        """
        self._token = token

    @property
    def token(self) -> str:
        return self._token

    def get_latest(self):
        with open('.cdc.tokens', 'rb') as fp:
            token = pickle.load(fp)
            return ResumeToken(token[0])


class PickledTokens:
    token_path = '.cdc.tokens'

    def __init__(self):
        self._current_tokens: t.List[str] = []

    def save(self, tokens: t.List[str] = []) -> None:
        with open(PickledTokens.token_path, 'wb') as fp:
            if tokens:
                pickle.dump(tokens, fp)
            else:
                pickle.dump(self._current_tokens, fp)

    def load(self) -> t.List[str]:
        with open(PickledTokens.token_path, 'rb') as fp:
            tokens = pickle.load(fp)
        self._current_tokens = tokens
        return tokens


class GenericDatabaseWatch(ABC):
    """GenericDatabaseWatch.
    A Base class which defines basic functions to implement.
    """

    indentity_sep = '/'

    @abstractmethod
    def watch(self):
        raise NotImplementedError

    @abstractmethod
    def attach_scheduler(self):
        raise NotImplementedError


class MongoEventMixin:
    """A Mixin class which defines helper fxns for `MongoDatabaseWatcher`
    It define basic events handling methods like
    `on_create`, `on_update`, etc.
    """

    DEFAULT_ID: str = '_id'
    EXCLUSION_KEYS: t.List[str] = [DEFAULT_ID]

    @staticmethod
    def submit_task_workflow(
        cdc_query: BaseModel, db: 'BaseDatabase', ids: t.List
    ) -> None:
        """submit_task_workflow.
        A fxn to build a taskflow and execute it with changed ids.
        This also extends the task workflow graph with a node.
        This node is responsible for applying a vector indexing watcher,
        and copying the vectors into a vector search database.

        :param cdc_query: A query which will be used by `db._build_task_workflow` method
        to extract the desired data.
        :type cdc_query: BaseModel
        :param db: a superduperdb instance.
        :type db: 'BaseDatabase'
        :param ids: List of ids which were observed as changed document.
        :type ids: t.List
        :rtype: None
        """

        task_graph = db._build_task_workflow(cdc_query, ids=ids, verbose=False)
        '''
        task_graph = MongoEventMixin.create_vector_watcher_task(
            task_graph, db=db, cdc_query=cdc_query
        )
        '''
        task_graph()

    # ruff: noqa: F821
    @staticmethod
    def create_vector_watcher_task(
        task_graph: TaskWorkflow, db: 'BaseDatabase', cdc_query: BaseModel
    ) -> TaskWorkflow:
        """create_vector_watcher_task.

        :param task_graph:
        :type task_graph: TaskWorkflow
        :param db:
        :type db: 'BaseDatabase'
        :param cdc_query:
        :type cdc_query: BaseModel
        :rtype: TaskWorkflow
        """
        for identifier in db.show('vector_index'):
            indexing_watcher = ...
            task_graph.add_node(
                f'copy_vectors({indexing_watcher})',
                FunctionJob(  # type: ignore
                    callable=copy_vectors,  # type: ignore
                    args=[identifier, cdc_query],
                    kwargs={},
                ),
            )
            model, key = indexing_watcher.split('/')  # type: ignore

            task_graph.add_edge(
                f'copy_vectors({indexing_watcher})',
                f'{model}.predict({key})',
            )
        return task_graph

    def on_create(
        self, change: t.Dict, db: 'BaseDatabase', collection: queries.Collection
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
        :type collection: queries.Collection
        :rtype: None
        """
        logging.debug('Triggerd `on_create` handler.')
        # new document added!
        document = change[CDCKeys.document_data_key]
        ids = [document[self.DEFAULT_ID]]
        query = collection.find()
        self.submit_task_workflow(query, db=db, ids=ids)

    def on_update(self, change: t.Dict, db: 'BaseDatabase'):
        """on_update.

        :param change:
        :type change: t.Dict
        :param db:
        :type db: 'BaseDatabase'
        """

        #  prepare updated document
        change[CDCKeys.update_descriptions_key]['updatedFields']
        change[CDCKeys.update_descriptions_key]['removedFields']


class CDCKeys(Enum):
    """
    A enum to represent mongo change document keys.
    """

    operation_type = 'operationType'
    document_key = 'documentKey'
    update_descriptions_key = 'updateDescription'
    document_data_key = 'fullDocument'


class MongoDatabaseWatcher(GenericDatabaseWatch, MongoEventMixin):
    """
    It is a class which helps capture data from mongodb database and handle it
    accordingly.

    This class accepts options and db instance from user and starts a scheduler
    which could schedule a watching service to watch change stream.

    This class builds a workflow graph on each change observed.

    """

    identity_sep: str = '/'

    def __init__(
        self,
        db: 'BaseDatabase',
        on: queries.Collection,
        identifier: 'str' = '',
        resume_token: t.Optional[ResumeToken] = None,
    ):
        """__init__.

        :param db: It is a superduperdb instance.
        :type db: 'BaseDatabase'
        :param on: It is used to define a Collection on which CDC would be performed.
        :type on: queries.Collection
        :param identifier: A identifier to represent the watcher service.
        :type identifier: 'str'
        :param resume_token: A resume token is a token used to resume
        the change stream in mongo.
        :type resume_token: t.Optional[ResumeToken]
        """
        self.db = db
        self._on_component = on
        self._identifier = self._build_identifier([identifier, on.name])
        self.tokens = PickledTokens()

        self.resume_token = resume_token.token if resume_token else None

        super().__init__()

    @property
    def identity(self) -> str:
        return self._identifier

    @classmethod
    def _build_identifier(cls, identifiers) -> str:
        """_build_identifier.

        :param identifiers: list of identifiers.
        :rtype: str
        """
        return cls.identity_sep.join(identifiers)

    @staticmethod
    def _get_stream_pipeline(
        change: t.Optional[t.Union[str, t.Dict]]
    ) -> t.Optional[t.Dict]:
        """_get_stream_pipeline.

        :param change: change can be a prebuilt watch pipeline like
        'generic' or user Defined watch pipeline.
        :type change: t.Optional[t.Union[str, t.Dict]]
        :rtype: t.Optional[t.Dict]
        """
        if not change:
            return MongoChangePipelines.get('generic')
        return None

    def _get_reference_id(self, document: t.Dict) -> t.Optional[str]:
        """_get_reference_id.

        :param document:
        :type document: t.Dict
        :rtype: t.Optional[str]
        """
        try:
            document_key = document[CDCKeys.document_key]
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
        event = change[CDCKeys.operation_type]
        reference_id = self._get_reference_id(change)

        if not reference_id:
            logging.warning('Document change not handled due to no document key')
            return

        if event == DBEvent.insert:
            self.on_create(change, db=self.db, collection=self._on_component)

        elif event == DBEvent.update:
            self.on_update(change, db=self.db)

    def dump_token(self, change: t.Dict) -> None:
        """dump_token.
        A helper utility to dump resume token from the changed document.

        :param change:
        :type change: t.Dict
        :rtype: None
        """
        token = change['_id']['_data']
        self.tokens.save([token])

    def cdc(self, change_pipeline: t.Optional[t.Union[str, t.Dict]] = None) -> None:
        """cdc.
        It is the primary entrypoint fxn to start cdc on the database and collection
        defined by the user.

        :param change:
        :type change: t.Optional[t.List]
        :rtype: None
        """

        try:
            pipeline = self._get_stream_pipeline(change_pipeline)

            stream = self._on_component.change_stream(
                pipeline=pipeline, resume_token=self.resume_token
            )

            stream_iterator = stream(self.db)
            logging.info(f'Started listening database with identity {self.identity}...')
            for change in stream_iterator:
                logging.debug(
                    f'Database change encountered at {datetime.datetime.now()}'
                )
                self.event_handler(change)
                self.dump_token(change)
        except Exception:
            logging.exception('Error occured during CDC!')
            raise

    def attach_scheduler(self, scheduler):
        self._scheduler = scheduler

    def watch(self, change_pipeline: t.Optional[t.Union[str, t.Dict]] = None) -> None:
        """Primary fxn to initiate watching of a database on the collection
        with defined `change_pipeline` by the user.

        :param change_pipeline:
        :rtype: None
        """
        if not self._scheduler:
            raise ValueError('No scheduler available!')

        try:
            self._scheduler.start()
        except Exception:
            logging.exception('Watching service stopped!')
            raise

    def close(self) -> None:
        """
        A placeholder fxn for future instances/session to be closed.
        Currently nothing to close.
        """
        ...
