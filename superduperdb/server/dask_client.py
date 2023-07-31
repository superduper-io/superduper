import typing as t
import uuid

from dask import distributed

from superduperdb import logging


class DaskClient:
    """
    A client for interacting with a Dask cluster.
    """

    def __init__(
        self,
        address: str,
        serializers: t.Optional[t.Sequence[t.Callable]] = None,
        deserializers: t.Optional[t.Sequence[t.Callable]] = None,
        local: bool = False,
        envs: t.Dict[str, t.Any] = {},
        **kwargs,
    ):
        """
        Initialize the DaskClient.

        :param address: The address of the Dask cluster.
        :param serializers: A list of serializers to be used by the client. (optional)
        :param local: Set to True to create a local Dask cluster. (optional)
        :param envs: An environment dict for cluster.
        """
        self.futures_collection: t.Dict[str, distributed.Future] = {}
        if local:
            cluster = distributed.LocalCluster(env=envs)
            self.client = distributed.Client(cluster, **kwargs)
        else:
            self.client = distributed.Client(
                address=address,
                serializers=serializers,
                deserializers=deserializers,
                **kwargs,
            )

    def submit(self, function: t.Callable, **kwargs) -> distributed.Future:
        """
        Submits a function to the Dask server for execution.

        :param function: The function to be executed.
        :param kwargs: Additional keyword arguments to be passed to the function.
        """
        future = self.client.submit(function, **kwargs)
        identifier = kwargs.get('identifier', None)
        if not identifier:
            logging.warn(
                'Could not get an identifier from submitted function, creating one!'
            )
            identifier = str(uuid.uuid4())
        self.futures_collection[identifier] = future
        return future

    def submit_and_forget(self, function: t.Callable, **kwargs) -> distributed.Future:
        """
        Submits a function to the Dask server and keep executing the future
        even if it is no longer referenced.

        :param function: The function to be executed.
        :param kwargs: Additional keyword arguments to be passed to the function.
        """
        future = self.submit(function, **kwargs)
        distributed.fire_and_forget(future)
        return future

    def shutdown(self) -> None:
        """
        Shuts down the Dask client.
        """
        self.client.shutdown()

    def wait_all_pending_tasks(self) -> None:
        """
        Waits for all pending tasks to complete.
        """
        futures = list(self.futures_collection.values())
        distributed.wait(futures)

    def get_result(self, identifier: str) -> t.Any:
        """
        Retrieves the result of a previously submitted task.
        Note: This will block until the future is completed.

        :param identifier: The identifier of the submitted task.
        """
        future = self.futures_collection[identifier]
        return self.client.gather(future)


def dask_client(
    cfg,
    envs: t.Dict[str, t.Any] = {},
    local: t.Optional[bool] = None,
    **kwargs,
) -> DaskClient:
    """
    Creates a DaskClient instance.

    :param cfg: Configuration object containing Dask cluster details.
    :param local: t.Set to True to create a local Dask cluster. (optional)
    :param envs: An environment dict for cluster.
    """
    return DaskClient(
        address=f'tcp://{cfg.ip}:{cfg.port}',
        serializers=cfg.serializers,
        deserializers=cfg.deserializers,
        envs=envs,
        local=local if local is not None else getattr(cfg, 'local', True),
        **kwargs,
    )
