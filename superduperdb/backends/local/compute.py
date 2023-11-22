import typing as t

from superduperdb import logging
from superduperdb.backends.base.compute import ComputeBackend


class LocalComputeBackend(ComputeBackend):
    """
    A mockup backend for running jobs locally.
    """

    def __init__(
        self,
    ):
        self.futures_collection: t.Dict[str, t.Any] = {}
        logging.info("Local compute engine is ready")

    def submit(self, function: t.Callable, **kwargs) -> None:
        """
        Submits a function for local execution.

        :param function: The function to be executed.
        :param kwargs: Additional keyword arguments to be passed to the function.
        """
        future = function(**kwargs)
        self.futures_collection[future.key] = future

        logging.success(f"Job submitted.  function:{function} future:{future}")
        return future

    def disconnect(self) -> None:
        """
        Disconnect the local client.
        """
        pass

    def shutdown(self) -> None:
        """
        Shuts down the local cluster.
        """
        pass

    def wait_all_pending_tasks(self) -> None:
        """
        Waits for all pending tasks to complete.
        """
        pass

    def get_result(self, identifier: str) -> t.Any:
        """
        Retrieves the result of a previously submitted task.
        Note: This will block until the future is completed.

        :param identifier: The identifier of the submitted task.
        """
        future = self.futures_collection[identifier]
        return future
