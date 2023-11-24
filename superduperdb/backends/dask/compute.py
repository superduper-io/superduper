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
        address: t.Optional[str] = None,
        serializers: t.Optional[t.Sequence[t.Callable]] = None,
        deserializers: t.Optional[t.Sequence[t.Callable]] = None,
        local: bool = False,
        **kwargs,
    ):
        # Private field
        self.__futures_collection: t.Dict[str, distributed.Future] = {}

        if local:
            # Create and connect to the local cluster.
            self.client = distributed.Client(processes=False)
        else:
            assert address, 'Address cannot be ``None`` for non local dask client'
            # Connect to a remote cluster.
            self.client = distributed.Client(
                address=address,
                serializers=serializers,
                deserializers=deserializers,
                **kwargs,
            )

    @property
    def type(self) -> str:
        return "distributed"

    @property
    def name(self) -> str:
        return "dask"

    def submit(self, function: t.Callable, *args, **kwargs) -> distributed.Future:
        """
        Submits a function to the Dask server for execution.

        :param function: The function to be executed.
        """
        future = self.client.submit(function, *args, **kwargs)
        self.__futures_collection[future.key] = future

        logging.success(f"Job submitted.  function:{function} future:{future}")
        return future

    def list_all_pending_tasks(self) -> t.Dict[str, distributed.Future]:
        """
        List for all pending tasks
        """
        return self.__futures_collection

    def wait_all_pending_tasks(self) -> None:
        """
        Waits for all pending tasks to complete.
        """
        futures = list(self.__futures_collection.values())
        distributed.wait(futures)

    def get_result(self, identifier: str) -> t.Any:
        """
        Retrieves the result of a previously submitted task.
        Note: This will block until the future is completed.

        :param identifier: The identifier of the submitted task.
        """
        future = self.__futures_collection[identifier]
        return self.client.gather(future)

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
