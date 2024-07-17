import typing as t
import uuid

from superduper import logging
from superduper.backends.base.compute import ComputeBackend
from superduper.jobs.queue import BaseQueuePublisher, LocalQueuePublisher


class LocalComputeBackend(ComputeBackend):
    """
    A mockup backend for running jobs locally.

    :param uri: Optional uri param.
    :param queue: Optional pluggable queue.
    """

    def __init__(
        self,
        uri: t.Optional[str] = None,
        queue: BaseQueuePublisher = LocalQueuePublisher(),
    ):
        self.__outputs: t.Dict = {}
        self.uri = uri
        self.queue = queue

    @property
    def remote(self) -> bool:
        """Return if remote compute engine."""
        return False

    @property
    def type(self) -> str:
        """The type of the backend."""
        return "local"

    @property
    def name(self) -> str:
        """The name of the backend."""
        return "local"

    def component_hook(self, *args, **kwargs):
        """Hook for component."""
        pass

    def broadcast(self, events: t.List):
        """Broadcast events to the corresponding component.

        :param events: List of events.
        :param to: Destination component.
        """
        return self.queue.publish(events)

    def submit(
        self, function: t.Callable, *args, compute_kwargs: t.Dict = {}, **kwargs
    ) -> t.Tuple[str, str]:
        """
        Submits a function for local execution.

        :param function: The function to be executed.
        :param args: Positional arguments to be passed to the function.
        :param compute_kwargs: Do not use this parameter.
        :param kwargs: Keyword arguments to be passed to the function.
        """
        logging.info(f"Submitting job. function:{function}")
        future = function(*args, **kwargs)

        future_key = str(uuid.uuid4())
        self.__outputs[future_key] = future

        logging.success(
            f"Job submitted on {self}.  function:{function} future:{future_key}"
        )
        return future_key, future_key

    @property
    def tasks(self) -> t.Dict[str, t.Any]:
        """List for all pending tasks."""
        return self.__outputs

    def wait_all(self) -> None:
        """Waits for all pending tasks to complete."""
        pass

    def result(self, identifier: str) -> t.Any:
        """Retrieves the result of a previously submitted task.

        Note: This will block until the future is completed.

        :param identifier: The identifier of the submitted task.
        """
        return self.__outputs[identifier]

    def disconnect(self) -> None:
        """Disconnect the local client."""
        pass

    def shutdown(self) -> None:
        """Shuts down the local cluster."""
        pass
