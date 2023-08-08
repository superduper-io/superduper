"""
The classes in this file define the configuration variables for SuperDuperDB,
which means that this file gets imported before alost anything else, and
canot contain any other imports from this project.
"""

import typing as t
from enum import Enum

from pydantic import Field, root_validator

from .jsonable import Factory, JSONable

_BAD_KEY = '...bad.key...'
REST_API_VERSION = '0.1.0'


class HasPort(JSONable):
    """A base class for server connections"""

    password: str = ''
    port: int = 0
    username: str = ''


class HostPort(HasPort):
    """Configure server connections with a `host` property"""

    host: str = 'localhost'


class IpPort(HasPort):
    """Configure server connections with an `ip` property"""

    ip: str = 'localhost'


class Retry(JSONable):
    """Describes how to retry using the `tenacity` library"""

    stop_after_attempt: int = 2
    wait_max: float = 10.0
    wait_min: float = 4.0
    wait_multiplier: float = 1.0


class Api(JSONable):
    """A base class for API connections"""

    api_key: str = Field(default=_BAD_KEY, repr=False)


class Apis(JSONable):
    """A container for API connections"""

    providers: t.Dict[str, Api] = Factory(dict)
    retry: Retry = Factory(Retry)


class Dask(IpPort):
    """Describes a connection to Dask"""

    deserializers: t.List[str] = Factory(list)
    port: int = 8786
    serializers: t.List[str] = Factory(list)
    local: bool = True


class Deployment(JSONable):
    """(unused)"""

    database: str = ''
    model: str = ''


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

    level: LogLevel = LogLevel.INFO
    type: LogType = LogType.STDERR
    kwargs: dict = Factory(dict)


class ModelServer(HostPort):
    """Configure the model server's port"""

    host: str = '127.0.0.1'
    port: int = 5001


class MongoDB(HostPort):
    """Configure MongoDB's port and credentials"""

    host: str = 'localhost'
    port: int = 27017


class DataLayer(JSONable):
    """Configure which db or database is being used"""

    cls: str = 'mongodb'
    connection: str = 'pymongo'
    kwargs: t.Dict = Factory(lambda: MongoDB().dict())
    name: str = 'test_db'


class DataLayers(JSONable):
    """TBD"""

    artifact: DataLayer = Factory(lambda: DataLayer(name='_filesystem:test_db'))
    data_backend: DataLayer = Factory(DataLayer)
    metadata: DataLayer = Factory(DataLayer)


class Notebook(JSONable):
    """Configure the notebook server connection information"""

    ip: str = '0.0.0.0'
    password: str = ''
    port: int = 8888
    token: str = ''

    @root_validator(skip_on_failure=True)  # type: ignore[call-overload]
    def check_one_or_none(cls, v: t.Dict) -> t.Dict:
        if v['password'] and v['token']:
            raise ValueError('At most one of password and token may be set')
        return v


class Server(JSONable):
    """Configure the SuperDuperDB server connection information"""

    host: str = '127.0.0.1'
    port: int = 3223
    protocol: str = 'http'

    @property
    def uri(self) -> str:
        return f'{self.protocol}://{self.host}:{self.port}'


class LanceDB(JSONable):
    """Configure the Lance DB vector search connection information"""

    backfill_batch_size: int = 100
    lancedb: bool = True
    uri: str = './.lancedb'


class InMemory(JSONable):
    """Configure the in-memory vector search connection information"""

    backfill_batch_size: int = 100
    inmemory: bool = True


class VectorSearch(JSONable):
    """Configure the full vector search connection information"""

    host: str = 'localhost'
    password: str = Field(default='', repr=False)
    port: int = 19530
    type: t.Union[LanceDB, InMemory] = Field(default_factory=InMemory)
    backfill_batch_size: int = 100
    username: str = Field(default='', repr=False)


class Downloads(JSONable):
    hybrid: bool = False
    root: str = 'data/downloads'


class Config(JSONable):
    """The data class containing all configurable superduperdb values"""

    apis: Apis = Factory(Apis)
    cdc: bool = False
    dask: Dask = Factory(Dask)
    data_layers: DataLayers = Factory(DataLayers)
    distributed: bool = False
    logging: Logging = Factory(Logging)
    model_server: ModelServer = Factory(ModelServer)
    notebook: Notebook = Factory(Notebook)
    server: Server = Factory(Server)
    vector_search: VectorSearch = Factory(VectorSearch)
    verbose: bool = False
    downloads: Downloads = Factory(Downloads)

    class Config(JSONable.Config):
        protected_namespaces = ()
