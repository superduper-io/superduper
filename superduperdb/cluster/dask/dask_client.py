import uuid

from dask.distributed import Client, wait, fire_and_forget

from superduperdb.misc.logger import logging


class DaskClient:
    def __init__(self, address: str, serializers=None, deserializers=None):
        self.futures_collection = {}
        self.client = Client(
            address=address,
            serializers=serializers,
            deserializers=deserializers,
        )

    def submit(self, function, **kwargs):
        future = self.client.submit(function, **kwargs)
        identifier = kwargs.get('identifier', None)
        if not identifier:
            logging.warning('Could not get an identifier from submitted function, creating one!')
            identifier = str(uuid.uuid4())
        self.futures_collection[identifier] = future
        return future

    def submit_and_forget(self, function, **kwargs):
        future = self.submit(function, **kwargs)
        fire_and_forget(future)
        return future

    def wait_all_pending_task(self):
        futures = self.futures_collection.values()
        wait(futures)

    def get_result(self, identifier: str):
        future = self.futures_collection[identifier]
        return self.client.gather(future)


def dask_client(cfg):
    return DaskClient(
        address=f'tcp://{cfg.ip}:{cfg.port}',
        serializers=cfg.serializers,
        deserializers=cfg.deserializers,
    )
