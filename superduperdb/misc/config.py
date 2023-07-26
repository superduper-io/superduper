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
    password: str = ''
    port: int = 0
    username: str = ''


class HostPort(HasPort):
    host: str = 'localhost'


class IpPort(HasPort):
    ip: str = 'localhost'


class Api(JSONable):
    api_key: str = Field(default=_BAD_KEY, repr=False)


class Retry(JSONable):
    stop_after_attempt: int = 2
    wait_max: float = 10.0
    wait_min: float = 4.0
    wait_multiplier: float = 1.0


class Apis(JSONable):
    providers: t.Dict[str, Api] = Factory(dict)
    retry: Retry = Factory(Retry)


class Dask(IpPort):
    deserializers: t.List[str] = Factory(list)
    port: int = 8786
    serializers: t.List[str] = Factory(list)
    local: bool = True


class Deployment(JSONable):
    database: str = ''
    model: str = ''


class LogLevel(str, Enum):
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARN = 'WARN'


class LogType(str, Enum):
    STDERR = 'STDERR'
    LOGGING = 'LOGGING'


class Logging(JSONable):
    level = LogLevel.INFO
    type = LogType.STDERR
    kwargs: dict = Factory(dict)


class ModelServer(HostPort):
    host: str = '127.0.0.1'
    port: int = 5001


class MongoDB(HostPort):
    host: str = 'localhost'
    password: str = 'testmongodbpassword'
    port: int = 27018
    username: str = 'testmongodbuser'


class DataLayer(JSONable):
    cls: str = 'mongodb'
    connection: str = 'pymongo'
    kwargs: t.Dict = Factory(lambda: MongoDB().dict())
    name: str = 'test_db'


class DataLayers(JSONable):
    artifact: DataLayer = Factory(lambda: DataLayer(name='_filesystem:test_db'))
    data_backend: DataLayer = Factory(DataLayer)
    metadata: DataLayer = Factory(DataLayer)


class Notebook(JSONable):
    ip: str = '0.0.0.0'
    password: str = ''
    port: int = 8888
    token: str = ''

    @root_validator
    def check_one_or_none(cls, v):
        if v['password'] and v['token']:
            raise ValueError('At most one of password and token may be set')
        return v


class Ray(HostPort):
    host: str = '127.0.0.1'

    deployments: t.List[Deployment] = Factory(list)


class Server(JSONable):
    host: str = '127.0.0.1'
    port: int = 3223
    protocol: str = 'http'

    @property
    def uri(self) -> str:
        return f'{self.protocol}://{self.host}:{self.port}'


class LanceDB(JSONable):
    backfill_batch_size: int = 100
    lancedb: bool = True
    uri: str = './.lancedb'


class InMemory(JSONable):
    backfill_batch_size: int = 100
    inmemory: bool = True


class VectorSearch(JSONable):
    host: str = 'localhost'
    password: str = Field(default='', repr=False)
    port: int = 19530
    type: t.Union[LanceDB, InMemory] = Field(default_factory=InMemory)
    username: str = Field(default='', repr=False)


class Config(JSONable):
    apis: Apis = Factory(Apis)
    cdc: bool = False
    dask: Dask = Factory(Dask)
    data_layers: DataLayers = Factory(DataLayers)
    distributed: bool = False
    logging: Logging = Factory(Logging)
    model_server: ModelServer = Factory(ModelServer)
    notebook: Notebook = Factory(Notebook)
    ray: Ray = Factory(Ray)
    server: Server = Factory(Server)
    vector_search: VectorSearch = Factory(VectorSearch)
