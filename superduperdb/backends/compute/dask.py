import typing as t

from dask import distributed

from superduperdb import logging
from superduperdb.backends.base.compute import ComputeBackend


class DaskComputeBackend(ComputeBackend):
    """
    A client for interacting with a Dask cluster. Initialize the DaskClient.

    :param address: The address of the Dask cluster.
    :param serializers: A list of serializers to be used by the client. (optional)
    :param deserializers: A list of deserializers to be used by the client. (optional)
    :param local: Set to True to create a local Dask cluster. (optional)
    :param envs: An environment dict for cluster.
    :param **kwargs: Additional keyword arguments to be passed to the DaskClient.
    """

    def __init__(
        self,
        address: str,
        serializers: t.Optional[t.Sequence[t.Callable]] = None,
        deserializers: t.Optional[t.Sequence[t.Callable]] = None,
        envs: t.Optional[t.Dict[str, t.Any]] = None,
        local: bool = False,
        **kwargs,
    ):
        envs = envs or {}
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

        logging.info("Compute Client is ready.", self.client)

    def submit(self, function: t.Callable, **kwargs) -> distributed.Future:
        """
        Submits a function to the Dask server for execution.

        :param function: The function to be executed.
        :param kwargs: Additional keyword arguments to be passed to the function.
        """
        future = self.client.submit(function, **kwargs)
        self.futures_collection[future.key] = future

        logging.success(f"Job submitted.  function:{function} future:{future}")
        return future

    def disconnect(self) -> None:
        """
        Disconnect the Dask client.
        """
        self.client.close()

    def shutdown(self) -> None:
        """
        Shuts down the Dask cluster.
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
