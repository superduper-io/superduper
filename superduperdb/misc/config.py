from .serializable import Serializable
from enum import Enum
from pydantic import Field, root_validator
from typing import Dict, List, Optional

"""
The classes in this file define the configuration variables for SuperDuperDB,
which means that this file gets imported before alost anything else, and
canot contain any other imports from this project.

There is a file in the root directory named `default-config.json`
which has the default values for every configuration variable, serialized into JSON.

If you change a class below, you must regenerate `default-config.json with

    $ python -m tests.unittests.misc.test_config
"""

_BAD_KEY = '...bad.key...'
REST_API_VERSION = '0.1.0'


def Factory(factory, **ka):
    return Field(default_factory=factory, **ka)


class HasPort(Serializable):
    port = 0
    password = ''
    user = ''


class HostPort(HasPort):
    host = 'localhost'


class IpPort(HasPort):
    ip = 'localhost'


class Api(Serializable):
    api_key: str = Field(default=_BAD_KEY, repr=False)


class Apis(Serializable):
    n_retries = 2
    providers: Dict[str, Api] = Factory(dict)


class Dask(IpPort):
    port = 8786

    serializers: List[str] = Factory(list)
    deserializers: List[str] = Factory(list)


class Deployment(Serializable):
    database = ''
    model = ''


class LogLevel(str, Enum):
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARN = 'WARN'


class LogType(str, Enum):
    STDERR = 'STDERR'
    LOGGING = 'LOGGING'


class Logging(Serializable):
    level = LogLevel.INFO
    type = LogType.STDERR
    kwargs: dict = Factory(dict)


class ModelServer(HostPort):
    host = '127.0.0.1'
    port = 5001


class MongoDB(HostPort):
    port = 27017


class Notebook(Serializable):
    ip = '0.0.0.0'
    port = 8888
    password = ''
    token = ''

    @root_validator
    def check_one_or_none(cls, v):
        assert not all(v.values()), 'At most one of password and token may be set'
        return v


class Ray(HostPort):
    host = '127.0.0.1'

    deployments: List[Deployment] = Factory(list)


class Server(Serializable):
    class FastAPI(Serializable):
        debug = False
        title = 'SuperDuperDB server'
        version = REST_API_VERSION

    class WebServer(Serializable):
        host = '127.0.0.1'
        port = 3223

    fastapi: FastAPI = Factory(FastAPI)
    web_server: WebServer = Factory(WebServer)


class MilvusConfig(Serializable):
    host: str = 'localhost'
    port: int = 19530
    username: str = Field(default="", repr=False)
    password: str = Field(default="", repr=False)
    db_name: str = "default"
    consistency_level: str = "Bounded"


class VectorSearchConfig(Serializable):
    milvus: Optional[MilvusConfig] = None

    # the fields below were left for compatibility with the vector search server
    # that is still in the codebase
    host: str = 'localhost'
    port: int = 5001
    user: str = Field(default="", repr=False)
    password: str = Field(default="", repr=False)


class Config(Serializable):
    apis: Apis = Factory(Apis)
    dask: Dask = Factory(Dask)
    logging: Logging = Factory(Logging)
    model_server: ModelServer = Factory(ModelServer)
    mongodb: MongoDB = Factory(MongoDB)
    notebook: Notebook = Factory(Notebook)
    ray: Ray = Factory(Ray)
    remote: bool = False
    server: Server = Factory(Server)
    vector_search: VectorSearchConfig = Factory(VectorSearchConfig)
