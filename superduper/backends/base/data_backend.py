import functools
import typing as t
from abc import ABC, abstractmethod

from superduper import logging
from superduper.backends.base.query import Query
from superduper.components.datatype import DataType

if t.TYPE_CHECKING:
    from superduper.components.schema import Schema


class BaseDataBackend(ABC):
    """Base data backend for the database.

    :param uri: URI to the databackend database.
    :param flavour: Flavour of the databackend.
    """

    db_type = None

    def __init__(self, uri: str, flavour: t.Optional[str] = None):
        self.conn = None
        self.name = 'base'
        self.flavour = flavour
        self.in_memory: bool = False
        self.in_memory_tables: t.Dict = {}
        self._datalayer = None
        self.uri = uri

    @property
    def type(self):
        """Return databackend."""
        raise NotImplementedError

    @property
    def db(self):
        """Return the datalayer."""
        raise NotImplementedError

    @abstractmethod
    def drop_outputs(self):
        """Drop all outputs."""

    @property
    def datalayer(self):
        """Return the datalayer."""
        return self._datalayer

    @datalayer.setter
    def datalayer(self, value):
        """Set the datalayer.

        :param value: The datalayer.
        """
        self._datalayer = value

    @abstractmethod
    def url(self):
        """Databackend connection url."""
        pass

    @abstractmethod
    def build_metadata(self):
        """Build a default metadata store based on current connection."""
        pass

    @abstractmethod
    def build_artifact_store(self):
        """Build a default artifact store based on current connection."""
        pass

    @abstractmethod
    def create_output_dest(
        self,
        predict_id: str,
        datatype: t.Union[str, DataType],
        flatten: bool = False,
    ):
        """Create an output destination for the database.

        :param predict_id: The predict id of the output destination.
        :param datatype: The datatype of the output destination.
        :param flatten: Whether to flatten the output destination.
        """
        pass

    @abstractmethod
    def create_table_and_schema(self, identifier: str, schema: "Schema"):
        """Create a schema in the data-backend.

        :param identifier: The identifier of the table.
        :param mapping: The mapping of the schema.
        """

    @abstractmethod
    def check_output_dest(self, predict_id) -> bool:
        """Check if the output destination exists.

        :param predict_id: The identifier of the output destination.
        """
        pass

    @abstractmethod
    def get_query_builder(self, key):
        """Get a query builder for the database.

        :param key: The key of the query builder,
                    typically the table or collection name.
        """
        pass

    @abstractmethod
    def get_table_or_collection(self, identifier):
        """Get a table or collection from the database.

        :param identifier: The identifier of the table or collection.
        """
        pass

    @abstractmethod
    def drop(self, force: bool = False):
        """Drop the databackend.

        :param force: If ``True``, don't ask for confirmation.
        """

    @abstractmethod
    def disconnect(self):
        """Disconnect the client."""

    @abstractmethod
    def list_tables_or_collections(self):
        """List all tables or collections in the database."""

    @staticmethod
    def infer_schema(data: t.Mapping[str, t.Any], identifier: t.Optional[str] = None):
        """Infer a schema from a given data object.

        :param data: The data object
        :param identifier: The identifier for the schema, if None, it will be generated
        :return: The inferred schema
        """

    def check_ready_ids(
        self, query: Query, keys: t.List[str], ids: t.Optional[t.List[t.Any]] = None
    ):
        """Check if all the keys are ready in the ids.

        :param query: The query object.
        :param keys: The keys to check.
        :param ids: The ids to check.
        """
        if ids:
            query = query.select_using_ids(ids)
        data = query.execute()
        ready_ids = []
        for select in data:
            notfound = 0
            for k in keys:
                try:
                    select[k]
                except KeyError:
                    notfound += 1
            if notfound == 0:
                ready_ids.append(select[query.primary_id])
        self._log_check_ready_ids_message(ids, ready_ids)
        return ready_ids

    def _log_check_ready_ids_message(self, input_ids, ready_ids):
        if input_ids and len(ready_ids) != len(input_ids):
            not_ready_ids = set(input_ids) - set(ready_ids)
            logging.info(f"IDs {not_ready_ids} do not ready.")
            logging.debug(f"Ready IDs: {ready_ids}")
            logging.debug(f"Not ready IDs: {not_ready_ids}")


class DataBackendProxy:
    """
    Proxy class to DataBackend which acts as middleware for performing fallbacks.

    :param backend: Instance of `BaseDataBackend`.
    """

    def __init__(self, backend):
        self._backend = backend

    @property
    def datalayer(self):
        """Return the datalayer."""
        return self._backend._datalayer

    @datalayer.setter
    def datalayer(self, value):
        """Set the datalayer.

        :param value: The datalayer.
        """
        self._backend._datalayer = value

    @property
    def type(self):
        """Instance of databackend."""
        return self._backend

    def _try_execute(self, attr):
        @functools.wraps(attr)
        def wrapper(*args, **kwargs):
            try:
                return attr(*args, **kwargs)
            except Exception as e:
                error_message = str(e).lower()
                if 'expire' in error_message and 'token' in error_message:
                    logging.warn(
                        f"Authentication expiry detected: {e}. "
                        "Attempting to reconnect..."
                    )
                    self._backend.reconnect()
                    return attr(*args, **kwargs)
                else:
                    raise e

        return wrapper

    def __getattr__(self, name):
        attr = getattr(self._backend, name)

        if callable(attr):
            return self._try_execute(attr)
        return attr
