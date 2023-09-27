import datetime
import threading
import traceback
import typing as t
from functools import cached_property

from overrides import override
from pymongo.change_stream import ChangeStream, CollectionChangeStream
from threa import Event, IsThread

from superduperdb import logging
from superduperdb.db.base.db import DB
from superduperdb.db.mongodb import CDC_COLLECTION_LOCKS, query

from .base import BaseDatabaseListener, CDCKeys, DBEvent, Packet, Tokens, TokenType
from .handler import CDC_QUEUE, CDCHandler

MONGO_CHANGE_PIPELINES: t.Dict[str, t.Sequence[t.Any]] = {'generic': []}
EVENT_TO_KEY = {
    DBEvent.delete: CDCKeys.deleted_document_data_key,
    DBEvent.insert: CDCKeys.document_data_key,
    DBEvent.update: CDCKeys.document_key,
}


class _DatabaseListenerThread(IsThread):
    looping = True

    def __init__(self, listener: BaseDatabaseListener):
        super().__init__()
        self.listener = listener

    @override
    def pre_run(self):
        self.cdc_stream = self.listener.setup_cdc()

    @override
    def callback(self) -> None:
        self.listener.next_cdc(self.cdc_stream)


class MongoDatabaseListener(BaseDatabaseListener):
    """
    It is a class which helps capture data from mongodb database and handle it
    accordingly.

    This class accepts options and db instance from user and starts a scheduler
    which could schedule a listening service to listen change stream.

    This class builds a workflow graph on each change observed.

    """

    DEFAULT_ID: str = '_id'
    EXCLUSION_KEYS: t.Sequence[str] = [DEFAULT_ID]
    IDENTITY_SEP: str = '/'

    _listener_thread: t.Optional[threading.Thread]
    _change_pipeline: t.Union[str, t.Sequence[t.Dict], None] = None

    def __init__(
        self,
        db: DB,
        on: query.Collection,
        identifier: 'str' = '',
        resume_token: t.Optional[TokenType] = None,
    ):
        """__init__.

        :param db: It is a superduperdb instance.
        :param on: It is used to define a Collection on which CDC would be performed.
        :param stopped: A threading event flag to notify for stoppage.
        :param identifier: A identifier to represent the listener service.
        :param resume_token: A resume token is a token used to resume
        the change stream in mongo.
        """
        self.db = db
        self._on_component = on
        self.identifier = f'{identifier}{self.IDENTITY_SEP}{on.name}'
        self.tokens = Tokens()
        self.info: t.Dict[str, int] = {'inserts': 0, 'updates': 0, 'deletes': 0}
        self.resume_token = resume_token

        self._change_pipeline = None
        self._listener_thread = None
        self._cdc_handler.start()

    @cached_property
    def _cdc_handler(self) -> CDCHandler:
        return CDCHandler(db=self.db)

    @property
    def running(self) -> Event:
        return self._listener_thread.running

    def event_handler(self, change: t.Dict) -> None:
        """event_handler.
        A helper fxn to handle incoming changes from change stream on a collection.

        :param change: The change (document) observed during the change stream.
        """
        event = change[CDCKeys.operation_type]
        try:
            document_key = change[CDCKeys.document_key]
            str(document_key['_id'])
        except KeyError:
            logging.warn('Document change not handled due to no document key')
            return

        self.info[event] = 1 + self.info.get(event, 0)
        logging.debug(f'Triggered {event} handler')

        document = change[EVENT_TO_KEY[event]]
        ids = [document[self.DEFAULT_ID]]

        query = None if event == DBEvent.delete else self._on_component.find()

        # TODO: Handle removed fields and updated fields.

        packet = Packet(ids=ids, event_type=event, query=query)
        CDC_QUEUE.put_nowait(packet)

    @override
    def setup_cdc(self) -> CollectionChangeStream:
        """
        Setup cdc change stream from user provided
        """
        if isinstance(self._change_pipeline, str):
            pipeline = MONGO_CHANGE_PIPELINES.get(self._change_pipeline)

        elif isinstance(self._change_pipeline, list):
            pipeline = self._change_pipeline or None

        else:
            raise TypeError(
                'Change pipeline can be either a string or a dictionary, '
                f'provided {type(self._change_pipeline)}'
            )
        stream = self._on_component.change_stream(
            pipeline=pipeline, resume_after=self.resume_token
        )

        logging.info(f'Started listening to database: {self.identifier}...')
        return stream(self.db)

    @override
    def next_cdc(self, stream: ChangeStream) -> None:
        """
        Get the next stream of change observed on the given `Collection`.
        """
        if (change := stream.try_next()) is not None:
            logging.debug(f'Database change encountered at {datetime.datetime.now()}')
            self.tokens.append(change[self.DEFAULT_ID])

            if change[CDCKeys.operation_type] == DBEvent.update:
                updates = change[CDCKeys.update_descriptions_key]
                updated_fields = updates[CDCKeys.update_field_key]
                if any(k.startswith('_outputs') for k in updated_fields):
                    return

            self.event_handler(change)

    @override
    def listen(
        self,
        change_pipeline: t.Optional[t.Union[str, t.Sequence[t.Dict]]] = None,
        timeout: t.Optional[float] = None,
    ) -> None:
        """Primary fxn to initiate listening of a database on the collection
        with defined `change_pipeline` by the user.

        :param change_pipeline: A mongo listen pipeline defined by the user
        for more fine grained listening.
        """
        try:
            CDC_COLLECTION_LOCKS[self._on_component.name] = True

            if self._listener_thread and self._listener_thread.is_alive():
                raise RuntimeError('CDC Listener thread is already running!')

            self._listener_thread = _DatabaseListenerThread(self)

            if change_pipeline is None:
                self._change_pipeline = MONGO_CHANGE_PIPELINES.get('generic')
            else:
                self._change_pipeline = change_pipeline

            self._listener_thread.start()
            self.running.wait(timeout=timeout)

        except Exception:
            logging.error('Listening service stopped!')
            CDC_COLLECTION_LOCKS.pop(self._on_component.name, None)

            self.stop()
            raise

    @override
    def stop(self) -> None:
        """
        Stop listening to cdc changes.
        This stops the corresponding services as well.
        """
        CDC_COLLECTION_LOCKS.pop(self._on_component.name, None)

        self._cdc_handler.stop()
        self._cdc_handler.join()

        self._listener_thread.stop()
        self._listener_thread.join()
