import typing as t
import uuid

from superduperdb import logging
from superduperdb.backends.base.compute import ComputeBackend


class LocalComputeBackend(ComputeBackend):
    """
    A mockup backend for running jobs locally.
    """

    def __init__(
        self,
    ):
        self.__outputs: t.Dict = {}

    @property
    def type(self) -> str:
        return "local"

    @property
    def name(self) -> str:
        return "local"

    def submit(self, function: t.Callable, *args, **kwargs) -> str:
        """
        Submits a function for local execution.

        :param function: The function to be executed.
        """
        logging.info(f"Submitting job. function:{function}")
        future = function(*args, **kwargs)

        future_key = str(uuid.uuid4())
        self.__outputs[future_key] = future

        logging.success(f"Job submitted.  function:{function} future:{future_key}")
        return future_key

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
