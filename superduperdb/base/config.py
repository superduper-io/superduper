"""
The classes in this file define the configuration variables for SuperDuperDB,
which means that this file gets imported before alost anything else, and
canot contain any other imports from this project.
"""

import typing as t
from enum import Enum

from pydantic import Field

from .jsonable import Factory, JSONable

_BAD_KEY = '...bad.key...'
REST_API_VERSION = '0.1.0'


class HasPort(JSONable):
    """
    A base class for server connections

    :param password: The password for the connection
    :param port: The port for the connection
    :param username: The username for the connection
    """

    password: str = ''
    port: int = 0
    username: str = ''


class HostPort(HasPort):
    """
    Configure server connections with a `host` property

    :param host: The host for the connection
    """

    host: str = 'localhost'


class IpPort(HasPort):
    """
    Configure server connections with an `ip` property

    :param ip: The ip for the connection
    """

    ip: str = 'localhost'


class Retry(JSONable):
    """
    Describes how to retry using the `tenacity` library

    :param stop_after_attempt: The number of attempts to make
    :param wait_max: The maximum time to wait between attempts
    :param wait_min: The minimum time to wait between attempts
    :param wait_multiplier: The multiplier for the wait time between attempts
    """

    stop_after_attempt: int = 2
    wait_max: float = 10.0
    wait_min: float = 4.0
    wait_multiplier: float = 1.0


class Api(JSONable):
    """
    A base class for API connections

    :param api_key: The API key for the connection
    """

    api_key: str = Field(default=_BAD_KEY, repr=False)


class Apis(JSONable):
    """
    A container for API connections

    :param providers: A dictionary of API connections
    :param retry: A ``Retry`` object
    """

    providers: t.Dict[str, Api] = Factory(dict)
    retry: Retry = Factory(Retry)


class Dask(IpPort):
    """
    Describes a connection to Dask

    :param deserializers: A list of deserializers
    :param port: The port for the connection
    :param serializers: A list of serializers
    :param local: Whether the connection is local
    """

    deserializers: t.List[str] = Factory(list)
    port: int = 8786
    serializers: t.List[str] = Factory(list)
    local: bool = True


class LogLevel(str, Enum):
    """Enumerate log severity level"""

    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARN = 'WARN'


class LogType(str, Enum):
    """Enumerate the standard logs"""

    STDERR = 'STDERR'
    LOGGING = 'LOGGING'


class Logging(JSONable):
    """
    Describe how we are going to log. This isn't yet used.

    :param level: The log level
    :param type: The log type
    :param kwargs: Any additional keyword arguments
    """

    level: LogLevel = LogLevel.INFO
    type: LogType = LogType.STDERR
    kwargs: dict = Factory(dict)


class MongoDB(HostPort):
    """
    Configure MongoDB's port and credentials

    :param host: The host for the connection
    :param port: The port for the connection
    """

    # TODO - configure all databases via uri

    host: str = 'localhost'
    port: int = 27017


class DbComponent(JSONable):
    """
    Configure which db or database is being used

    :param cls: The class of the data layer
    :param connection: The connection for the data layer
    :param kwargs: Any additional keyword arguments
    :param name: The name of the data layer
    """

    cls: str = 'ibis'
    connection: str = 'ibis'
    kwargs: t.Dict = Factory(lambda: MongoDB().dict())
    name: str = 'test_db'


class DbComponents(JSONable):
    """
    A container for the components of ``DB``.

    :param artifact: The artifact data layer
    :param data_backend: The data backend data layer
    :param metadata: The metadata data layer
    """
    artifact: DbComponent = Factory(lambda: DbComponent(name='_filesystem:test_db'))
    data_backend: DbComponent = Factory(DbComponent)
    metadata: DbComponent = Factory(DbComponent)


class Server(JSONable):
    """
    Configure the SuperDuperDB server connection information

    :param host: The host for the connection
    :param port: The port for the connection
    :param protocol: The protocol for the connection
    """

    host: str = '127.0.0.1'
    port: int = 3223
    protocol: str = 'http'

    @property
    def uri(self) -> str:
        return f'{self.protocol}://{self.host}:{self.port}'


class LanceDB(JSONable):
    """
    Configure the Lance DB vector search connection information

    :param backfill_batch_size: The batch size for backfilling
    :param lancedb: Whether LanceDB is being used
    :param uri: The URI for the connection
    """

    backfill_batch_size: int = 100
    lancedb: bool = True
    uri: str = './.lancedb'


class InMemory(JSONable):
    """
    Configure the in-memory vector search connection information

    :param backfill_batch_size: The batch size for backfilling
    :param inmemory: Whether in-memory is being used
    """

    backfill_batch_size: int = 100
    inmemory: bool = True


class SelfHosted(JSONable):
    """A placeholder for the case in which search is self-hosted by data-backend

    For example: MongoDB atlas vector-search, Elastic search etc..

    :param selfhosted: Whether self-hosted is being used
    """

    selfhosted: bool = True


class VectorSearch(JSONable):
    """
    Configure the full vector search connection information

    :param host: The host for the connection
    :param password: The password for the connection
    :param port: The port for the connection
    :param type: The type of vector search
    :param backfill_batch_size: The batch size for backfilling
    :param username: The username for the connection
    """

    host: str = 'localhost'
    password: str = Field(default='', repr=False)
    port: int = 19530
    type: t.Union[LanceDB, InMemory, SelfHosted] = Field(default_factory=InMemory)
    backfill_batch_size: int = 100  # TODO don't need this as well
    username: str = Field(default='', repr=False)


class Downloads(JSONable):
    """
    Configure how downloads are saved in the database
    or to hybrid filestorage (references to filesystem from datastore)

    :param hybrid: Whether hybrid is being used
    :param root: The root for the connection
    """

    hybrid: bool = False
    root: str = 'data/downloads'


class Config(JSONable):
    """
    The data class containing all configurable superduperdb values

    :param apis: The ``Apis`` object
    :param cdc: Whether CDC is being used
    :param dask: The ``Dask`` object
    :param data_layers: The ``DBComponents`` object
    :param distributed: Whether distributed is being used
    :param logging: The ``Logging`` object
    :param server: The ``Server`` object
    :param vector_search: The ``VectorSearch`` object
    :param downloads: The ``Downloads`` object
    :param verbose: Whether verbose is being used
    """

    apis: Apis = Factory(Apis)
    cdc: bool = False
    dask: Dask = Factory(Dask)
    db_components: DbComponents = Factory(DbComponents)
    distributed: bool = False  # TODO include this in the dask component
    logging: Logging = Factory(Logging)
    server: Server = Factory(Server)
    vector_search: VectorSearch = Factory(VectorSearch)
    downloads: Downloads = Factory(Downloads)

    class Config(JSONable.Config):
        protected_namespaces = ()
