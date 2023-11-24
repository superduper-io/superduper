import typing as t
from abc import ABC, abstractmethod, abstractproperty


class ComputeBackend(ABC):
    """
    Abstraction for sending jobs to a distributed compute platform.
    """

    @abstractproperty
    def type(self) -> str:
        """
        Return the type of compute engine
        """
        pass

    @abstractproperty
    def name(self) -> str:
        """
        Return the name of current compute engine
        """
        pass

    def get_local_client(self):
        '''Returns a local version of self'''
        pass

    @abstractmethod
    def submit(self, function: t.Callable, **kwargs) -> t.Any:
        """
        Submits a function to the Dask server for execution.

        :param function: The function to be executed.
        :param kwargs: Additional keyword arguments to be passed to the function.
        """
        pass

    @abstractmethod
    def list_all_pending_tasks(self) -> t.Any:
        """
        List for all pending tasks
        """
        pass

    @abstractmethod
    def wait_all_pending_tasks(self) -> None:
        """
        Waits for all pending tasks to complete.
        """
        pass

    @abstractmethod
    def get_result(self, identifier: str) -> t.Any:
        """
        Retrieves the result of a previously submitted task.
        Note: This will block until the future is completed.

        :param identifier: The identifier of the submitted task.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Disconnect the client.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        Shuts down the compute cluster.
        """
        pass
