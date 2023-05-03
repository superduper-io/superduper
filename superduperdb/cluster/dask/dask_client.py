from dask.distributed import Client
from superduperdb import cf


def dask_client():
    return Client(address=f'tcp://{cf["dask"]["ip"]}:{cf["dask"]["port"]}',
                  serializers=cf["dask"]["serializers"],
                  deserializers=cf["dask"]["deserializers"])
