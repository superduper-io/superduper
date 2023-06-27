from dask.distributed import Client
from superduperdb import CFG


def dask_client() -> Client:
    return Client(
        address=f'tcp://{CFG.dask.ip}:{CFG.dask.port}',
        serializers=CFG.dask.serializers,
        deserializers=CFG.dask.deserializers,
    )
