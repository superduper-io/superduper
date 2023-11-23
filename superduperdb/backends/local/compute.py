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
        self.__outputs: t.Dict[str, t.Any] = {}

    def name(self) -> str:
        return "Local"

    def submit(self, function: t.Callable, **kwargs) -> None:
        """
        Submits a function for local execution.

        :param function: The function to be executed.
        :param kwargs: Additional keyword arguments to be passed to the function.
        """
        logging.info(f"Submitting job. function:{function}")
        future = function(**kwargs)
        self.__outputs[future.key] = future

        logging.success(f"Job submitted.  function:{function} future:{future}")
        return future

    def list_all_pending_tasks(self) -> t.Dict[str, t.Any]:
        """
        List for all pending tasks
        """
        return self.__outputs

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
        return self.__outputs[identifier]

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
