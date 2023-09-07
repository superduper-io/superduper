"""
The classes in this file define the configuration variables for SuperDuperDB,
which means that this file gets imported before alost anything else, and
canot contain any other imports from this project.
"""

import typing as t
from enum import Enum

from .jsonable import Factory, JSONable


class Retry(JSONable):
    """Describes how to retry using the `tenacity` library"""

    #: The number of attempts to make
    stop_after_attempt: int = 2

    #: The maximum time to wait between attempts
    wait_max: float = 10.0

    #: The minimum time to wait between attempts
    wait_min: float = 4.0

    #: The multiplier for the wait time between attempts
    wait_multiplier: float = 1.0


class Apis(JSONable):
    """A container for API connections"""

    #: A ``Retry`` object
    retry: Retry = Factory(Retry)


class Cluster(JSONable):
    """Describes a connection to distributed work via Dask"""

    #: Whether to use distributed task management via Dask or not
    distributed: bool = False

    #: A list of deserializers
    deserializers: t.List[str] = Factory(list)

    #: A list of serializers
    serializers: t.List[str] = Factory(list)

    #: The Dask scheduler URI
    dask_scheduler: str = 'tcp://localhost:8786'

    #: Whether the connection is local
    local: bool = True
    backfill_batch_size: int = 100


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
    """Describe how we are going to log. This isn't yet used."""

    #: The log level
    level: LogLevel = LogLevel.INFO

    #: The log type
    type: LogType = LogType.STDERR

    #: Any additional keyword arguments
    kwargs: dict = Factory(dict)


class Server(JSONable):
    """Configure the SuperDuperDB server connection information"""

    #: The host for the connection
    host: str = '127.0.0.1'

    #: The port for the connection
    port: int = 3223

    #: The protocol for the connection
    protocol: str = 'http'

    @property
    def uri(self) -> str:
        return f'{self.protocol}://{self.host}:{self.port}'


class Downloads(JSONable):
    """
    Configure how downloads are saved in the database
    or to hybrid filestorage (references to filesystem from datastore)
    """

    #: Whether hybrid is being used
    hybrid: bool = False

    #: The root for the connection
    root: str = 'data/downloads'


class Config(JSONable):
    """The data class containing all configurable superduperdb values"""

    # 4 main components are URI strings
    #: The URI for the data backend
    data_backend: str = 'mongodb://localhost:27017'

    #: The URI for the vector search
    vector_search: 'str' = 'inmemory://'  # 'lancedb://./.lancedb'

    #: The URI for the artifact store
    artifact_store: t.Optional[str] = None

    #: The URI for the metadata store
    metadata_store: t.Optional[str] = None

    #: Settings distributed computing and change data capture
    cluster: Cluster = Factory(Cluster)

    #: Settings for OPENAI and other APIs
    apis: Apis = Factory(Apis)

    #: Logging
    logging: Logging = Factory(Logging)

    #: Settings for the experimental Rest server
    server: Server = Factory(Server)

    #: Settings for downloading files
    downloads: Downloads = Factory(Downloads)

    class Config(JSONable.Config):
        protected_namespaces = ()
