import pickle
import typing as T
from abc import ABC, abstractmethod
from dataclasses import dataclass

from superduperdb.misc.logger import logging as logger
from superduperdb.queries.mongodb.queries import Collection

MongoChangePipelines = {'generic': []}


@dataclass
class DBEvent:
    insert = 'insert'
    update = 'update'
    delete = 'delete'


class MongoChangePipeline:
    def __init__(
        self,
    ):
        self.valid_ops: T.List[T.Text] = [
            DBEvent.insert,
            DBEvent.update,
            DBEvent.delete,
        ]

    def validate(self):
        raise NotImplementedError

    def build_matching(self, matching_operations: T.List = []):
        if any([op for op in matching_operations if op not in self.valid_ops]):
            raise ValueError('not valid ops')

        pipeline = {'$match': {'operationType': {'$in': [*matching_operations]}}}
        return pipeline


class ResumeToken:
    def __init__(self, token=''):
        self._token = token

    @property
    def token(self):
        return self._token

    def get_latest(self):
        with open('.cdc.tokens', 'rb') as fp:
            token = pickle.load(fp)
            return ResumeToken(token[0])


class PickledTokens:
    NAME = '.cdc.tokens'

    def __init__(self):
        self._current_tokens = []

    def save(self, tokens: list = []):
        with open(self.NAME, 'wb') as fp:
            if tokens:
                pickle.dump(tokens, fp)
            else:
                pickle.dump(self._current_tokens, fp)

    def load(self):
        with open(self.NAME, 'rb') as fp:
            tokens = pickle.load(fp)
        self._current_tokens = tokens
        return tokens


class GenericDatabaseWatch(ABC):
    indentity_sep = "/"

    def __init__(self):
        ...

    @abstractmethod
    def watch(self):
        raise NotImplementedError

    @abstractmethod
    def attach_scheduler(self):
        raise NotImplementedError


class MongoEventMixin:
    _document_data_key = "documentKey"
    _update_descriptions_key = "updateDescription"
    exclusion_key = ["_id"]

    def on_create(self, change, reference_id, db=None):
        # new document added!
        document = change[MongoEventMixin._document_data_key]
        logger.info("Triggerd on_create handler.")

        cdc_data = {k: v for k, v in document.items() if k not in self.exclusion_key}
        G = db.refresh_after_update_or_insert()
        G()

    def on_update(self, change, reference_id, db=None):

        #  prepare updated document
        updated_fields = change[MongoEventMixin._update_descriptions_key][
            'updatedFields'
        ]
        removed_fields = change[MongoEventMixin._update_descriptions_key][
            'removedFields'
        ]

    def on_delete(self, change, reference_id, db=None):
        #  prepare updated document
        # TODO (low) : do we need Packet to send it to queue
        # packet = Packet(event_type=DBEvent.delete, reference_id=reference_id)
        ...


class MongoDatabaseWatcher(GenericDatabaseWatch, MongoEventMixin):
    identity_sep = '/'

    def __init__(self, db, on: Collection, resume_token=None):
        self.db = db
        self._on_component = on
        self.__identifier = self._build_identifier([db.identifier, on.name])
        self.tokens = PickledTokens()

        self.resume_token = resume_token.token if resume_token else None
        self._operation_type = "operationType"
        self.exclusion_key = ["_id"]
        self._document_data_key = "fullDocument"
        self.document_key = "documentKey"
        self.update_descriptions_key = "updateDescription"

        super().__init__()

    @property
    def indentity(self):
        return self.__identifier

    @classmethod
    def _build_identifier(cls, identifiers):
        return cls.identity_sep.join(*identifiers)

    @staticmethod
    def _get_stream_pipeline(change):
        if not change:
            return MongoChangePipelines.get('generic')

    def _get_reference_id(self, document):
        try:
            document_key = document[self.document_key]
            reference_id = str(document_key['_id'])
        except KeyError:
            return None
        return reference_id

    def event_handler(self, change):
        event = change[self._operation_type]
        reference_id = self._get_reference_id(change)
        if not reference_id:
            logger.warning("Document change not handled due to no document key")
            return

        if event == DBEvent.insert:
            self.on_create(change, reference_id, db=self.db)

        elif event == DBEvent.update:
            logger.info("Triggerd on_update handler.")
            self.on_update(change, reference_id, db=self.db)

        elif event == DBEvent.delete:
            logger.info("Triggerd on_delete handler.")
            self.on_delete(change, reference_id, db=self.db)

    def dump_token(self, change):
        token = change['_id']['_data']
        self.tokens.save([token])

    def cdc(self, change=None):

        pipeline = self._get_stream_pipeline(change)

        _stream = self._on_component.change_stream(
            pipeline=pipeline, resume_token=self.resume_token
        )

        try:
            logger.info("started listening database...")
            for change in _stream(self.db):
                logger.info("database change encountered.")
                self.event_handler(change)
                self.dump_token(change)
        except Exception as exc:
            logger.exception("Error occured during CDC!")
            raise

    def attach_scheduler(self, scheduler):
        self._scheduler = scheduler

    def watch(self, change=None):
        if not self._scheduler:
            raise ValueError("No scheduler available!")

        try:
            self._scheduler.start()
        except Exception as exc:
            logger.exception("Watching service stopped!")
            raise

    def close(self):
        self.client.close()
