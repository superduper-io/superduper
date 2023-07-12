"""
The classes in this file define the configuration variables for SuperDuperDB,
which means that this file gets imported before alost anything else, and
canot contain any other imports from this project.

There is a file in the root directory named `default-config.json`
which has the default values for every configuration variable, serialized into JSON.

If you change a class below, you must regenerate `default-config.json with

    $ python -m tests.unittests.misc.test_config
"""

from .jsonable import JSONable, Factory
from enum import Enum
from pydantic import Field, root_validator
import typing as t


_BAD_KEY = '...bad.key...'
REST_API_VERSION = '0.1.0'


class HasPort(JSONable):
    port: int = 0
    password: str = ''
    username: str = ''


class HostPort(HasPort):
    host: str = 'localhost'


class IpPort(HasPort):
    ip: str = 'localhost'


class Api(JSONable):
    api_key: str = Field(default=_BAD_KEY, repr=False)


class Retry(JSONable):
    wait_multiplier: float = 1.0
    wait_min: float = 4.0
    wait_max: float = 10.0

    stop_after_attempt: int = 2


class Apis(JSONable):
    providers: t.Dict[str, Api] = Factory(dict)
    retry: Retry = Factory(Retry)


class Dask(IpPort):
    port: int = 8786

    serializers: t.List[str] = Factory(list)
    deserializers: t.List[str] = Factory(list)


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
    host: str = "localhost"
    port: int = 27018
    username: str = "testmongodbuser"
    password: str = "testmongodbpassword"


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
    port: int = 8888
    password: str = ''
    token: str = ''

    @root_validator
    def check_one_or_none(cls, v):
        assert not all(v.values()), 'At most one of password and token may be set'
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
    lancedb: bool = True
    uri: str = './.lancedb'
    backfill_batch_size: int = 100


class InMemory(JSONable):
    inmemory: bool = True
    uri: str = ''
    backfill_batch_size: int = 100


class Milvus(JSONable):
    milvus: bool = True
    host: str = 'localhost'
    port: int = 19530
    username: str = Field(default='', repr=False)
    password: str = Field(default='', repr=False)
    db_name: str = 'default'
    consistency_level: str = 'Bounded'
    backfill_batch_size: int = 100


class VectorSearch(JSONable):
    type: t.Union[LanceDB, InMemory, Milvus] = Field(default_factory=InMemory)
    username: str = Field(default='', repr=False)
    password: str = Field(default='', repr=False)
    host: str = 'localhost'
    port: int = 19530


class Config(JSONable):
    apis: Apis = Factory(Apis)
    dask: Dask = Factory(Dask)
    logging: Logging = Factory(Logging)
    model_server: ModelServer = Factory(ModelServer)
    data_layers: DataLayers = Factory(DataLayers)
    notebook: Notebook = Factory(Notebook)
    ray: Ray = Factory(Ray)
    distributed: bool = False
    cdc: bool = False
    server: Server = Factory(Server)
    vector_search: VectorSearch = Factory(VectorSearch)
