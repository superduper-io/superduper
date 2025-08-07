import dataclasses as dc
import datetime
import threading
import typing as t
from enum import Enum

from pymongo.change_stream import CollectionChangeStream
from superduper import CFG, logging
from superduper.backends.base.cdc import BaseDatabaseListener, DBEvent

if t.TYPE_CHECKING:
    pass


class CDCKeys(str, Enum):
    """A enum to represent mongo change document keys."""

    operation_type = "operationType"
    document_key = "documentKey"
    update_descriptions_key = "updateDescription"
    update_field_key = "updatedFields"
    document_data_key = "fullDocument"
    deleted_document_data_key = "documentKey"


_CDCKEY_MAP = {
    DBEvent.update: CDCKeys.document_key,
    DBEvent.insert: CDCKeys.document_data_key,
    DBEvent.delete: CDCKeys.deleted_document_data_key,
}


@dc.dataclass
class ChangeStream:
    """Change stream class to watch for changes in specified collection.

    :param collection: The collection to perform the query on
    :param args: Positional query arguments to ``pymongo.Collection.watch``
    :param kwargs: Named query arguments to ``pymongo.Collection.watch``
    """

    collection: str
    args: t.Sequence = dc.field(default_factory=list)
    kwargs: t.Dict = dc.field(default_factory=dict)

    def __call__(self, db):
        """Watch for changes in the database in specified collection.

        :param db: The datalayer instance
        """
        db_name = CFG.data_backend.split('/')[-1]
        collection = db.databackend.conn[db_name][self.collection]
        return collection.watch(**self.kwargs)


class MongoDBDatabaseListener(BaseDatabaseListener):
    """A class handling change stream in mongodb.

    It is a class which helps capture data from mongodb database and handle it
    accordingly.

    This class accepts options and db instance from user and starts a scheduler
    which could schedule a listening service to listen change stream.

    This class builds a workflow graph on each change observed.

    :param db: It is a datalayer instance.
    :param on: It is used to define a Collection on which CDC would be performed.
    :param stop_event: A threading event flag to notify for stoppage.
    :param identifier: A identifier to represent the listener service.
    :param timeout: A timeout to stop the listener service.
    """

    _scheduler: t.Optional[threading.Thread]

    def _get_reference_id(self, document: t.Dict) -> t.Optional[str]:
        """_get_reference_id.

        :param document:
        """
        try:
            document_key = document[CDCKeys.document_key]
            reference_id = str(document_key["_id"])
        except KeyError:
            return None
        return reference_id

    def setup_cdc(self) -> CollectionChangeStream:
        """Setup cdc change stream from user provided."""
        stream = ChangeStream(
            collection=self.table,
        )

        stream_iterator = stream(self.db)

        logging.info(f"Started listening to MongoDB change stream on: {self.table}")

        return stream_iterator

    def next_cdc(self, stream: CollectionChangeStream) -> None:
        """Get the next stream of change observed on the given `Collection`.

        :param stream: A change stream object.
        """
        change = stream.try_next()
        if change is not None:
            logging.debug(f"Database change encountered at {datetime.datetime.now()}")

            reference_id = self._get_reference_id(change)

            if not reference_id:
                logging.warn("Document change not handled due to no document key")
                return

            event = change[CDCKeys.operation_type]
            ids = [change[_CDCKEY_MAP[event]]['_id']]
            self.event_handler(ids, event)
