import uuid

from dask.distributed import LocalCluster
from dask.distributed import Client, wait, fire_and_forget

from superduperdb.misc.logger import logging


class DaskClient:
    def __init__(self, address: str, serializers=None, deserializers=None, local=False):
        self.futures_collection = {}
        if local:
            cluster = LocalCluster(n_workers=1)
            self.client = Client(cluster)
        else:
            self.client = Client(
                address=address,
                serializers=serializers,
                deserializers=deserializers,
            )

    def submit(self, function, **kwargs):
        future = self.client.submit(function, **kwargs)
        identifier = kwargs.get('identifier', None)
        if not identifier:
            logging.warning(
                'Could not get an identifier from submitted function, creating one!'
            )
            identifier = str(uuid.uuid4())
        self.futures_collection[identifier] = future
        return future

    def submit_and_forget(self, function, **kwargs):
        future = self.submit(function, **kwargs)
        fire_and_forget(future)
        return future

    def shutdown(self):
        self.client.shutdown()

    def wait_all_pending_tasks(self):
        futures = list(self.futures_collection.values())
        wait(futures)

    def get_result(self, identifier: str):
        future = self.futures_collection[identifier]
        return self.client.gather(future)


def dask_client(cfg, local=False):
    return DaskClient(
        address=f'tcp://{cfg.ip}:{cfg.port}',
        serializers=cfg.serializers,
        deserializers=cfg.deserializers,
        local=local,
    )
