from pydantic import BaseModel, Field
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


class Port(_Model):
    port = 0


class HostPort(Port):
    host = 'localhost'


class IpPort(Port):
    ip = 'localhost'


class Api(_Model):
    api_key: str = Field(default=_BAD_KEY, repr=False)


class Dask(IpPort):
    port = 8786

    serializers: List[str] = _Factory(list)
    deserializers: List[str] = _Factory(list)


class Deployment(_Model):
    database = ''
    model = ''


class ModelServer(HostPort):
    host = '127.0.0.1'
    port = 5001


class MongoDB(HostPort):
    port = 27017


class Notebook(IpPort):
    port = 8888
    ip = '0.0.0.0'
    token: str = Field(default=_BAD_KEY, repr=False)


class Ray(HostPort):
    host = '127.0.0.1'

    deployments: List[Deployment] = _Factory(list)


class VectorSearch(HostPort):
    port = 5001


class Config(_Model):
    apis: Dict[str, Api] = _Factory(dict)
    dask: Dask = _Factory(Dask)
    model_server: ModelServer = _Factory(ModelServer)
    mongodb: MongoDB = _Factory(MongoDB)
    notebook: Notebook = _Factory(Notebook)
    ray: Ray = _Factory(Ray)
    remote: bool = False
    vector_search: VectorSearch = _Factory(VectorSearch)
