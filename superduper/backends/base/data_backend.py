import functools
import typing as t
from abc import ABC, abstractmethod

from superduper import logging
from superduper.backends.base.query import Query
from superduper.components.datatype import BaseDataType

if t.TYPE_CHECKING:
    from superduper.components.schema import Schema


class BaseDataBackend(ABC):
    """Base data backend for the database.

    :param uri: URI to the databackend database.
    :param plugin: Plugin implementing the databackend.
    :param flavour: Flavour of the databackend.
    """

    db_type = None

    def __init__(self, uri: str, plugin: t.Any, flavour: t.Optional[str] = None):
        self.conn = None
        self.flavour = flavour
        self.in_memory: bool = False
        self.in_memory_tables: t.Dict = {}
        self.plugin = plugin
        self._datalayer = None
        self.uri = uri
        self.bytes_encoding = 'bytes'

    @property
    def backend_name(self):
        return self.__class__.__name__.split('DataBackend')[0].lower()

    @property
    def type(self):
        """Return databackend."""
        raise NotImplementedError

    @property
    def db(self):
        """Return the DB."""
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
        datatype: t.Union[str, BaseDataType],
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

    def get_query_builder(self, key):
        """Get a query builder for the database.

        :param key: The key of the query builder,
                    typically the table or collection name.
        """
        return self.plugin.Query(table=key, db=self.datalayer)

    @abstractmethod
    def get_table(self, identifier):
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
    def list_tables(self):
        """List all tables or collections in the database."""

    @abstractmethod
    def reconnect(self):
        """Reconnect to the databackend"""


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
