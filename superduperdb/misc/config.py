from enum import Enum, auto
from pydantic import BaseModel, Field, root_validator
from typing import Dict, List

# The classes in this file define the configuration variables for SuperDuperDB.
#
# There is a file in the root directory named `default-config.json`
# which has the default values for every configuration variable, serialized into JSON.
#
# If you change a class below, you must regenerate `default-config.json with:
#
#    $ python -m tests.unittests.misc.test_config

_BAD_KEY = '...bad.key...'


def _Factory(factory):
    return Field(default_factory=factory)


class _Model(BaseModel):
    class Config:
        extra = 'forbid'


class HasPort(_Model):
    port = 0
    password = ''
    user = ''


class HostPort(HasPort):
    host = 'localhost'


class IpPort(HasPort):
    ip = 'localhost'


class Api(_Model):
    api_key: str = Field(default=_BAD_KEY, repr=False)


class Apis(_Model):
    n_retries = 2
    providers: Dict[str, Api] = _Factory(dict)


class Dask(IpPort):
    port = 8786

    serializers: List[str] = _Factory(list)
    deserializers: List[str] = _Factory(list)


class Deployment(_Model):
    database = ''
    model = ''


class LogLevel(str, Enum):
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARN = 'WARN'


class LogType(str, Enum):
    STDERR = 'STDERR'
    LOGGING = 'LOGGING'


class Logging(_Model):
    level = LogLevel.INFO
    type = LogType.STDERR
    kwargs: dict = _Factory(dict)


class ModelServer(HostPort):
    host = '127.0.0.1'
    port = 5001


class MongoDB(HostPort):
    port = 27017


class Notebook(_Model):
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

    deployments: List[Deployment] = _Factory(list)


class VectorSearch(HostPort):
    port = 5001


class Config(_Model):
    apis: Apis = _Factory(Apis)
    dask: Dask = _Factory(Dask)
    logging: Logging = _Factory(Logging)
    model_server: ModelServer = _Factory(ModelServer)
    mongodb: MongoDB = _Factory(MongoDB)
    notebook: Notebook = _Factory(Notebook)
    ray: Ray = _Factory(Ray)
    remote: bool = False
    vector_search: VectorSearch = _Factory(VectorSearch)
