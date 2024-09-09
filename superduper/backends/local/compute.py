import typing as t
import uuid

from superduper import logging
from superduper.backends.base.compute import ComputeBackend
from superduper.jobs.job import Job
from superduper.jobs.queue import BaseQueuePublisher, LocalQueuePublisher


class LocalComputeBackend(ComputeBackend):
    """
    A mockup backend for running jobs locally.

    :param uri: Optional uri param.
    :param queue: Optional pluggable queue.
    :param kwargs: Optional kwargs.
    """

    def __init__(
        self,
        uri: t.Optional[str] = None,
        queue: BaseQueuePublisher = LocalQueuePublisher(),
        **kwargs,
    ):
        self.uri = uri
        self.queue = queue
        self.kwargs = kwargs
        self._cache = {}

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

    def declare_component(self, component: t.Any) -> None:
        self._cache[component.type_id, component.identifier, component.uuid] = component

    def component_hook(self, *args, **kwargs):
        """Hook for component."""
        pass

    def broadcast(self, events: t.List):
        """Broadcast events to the corresponding component.

        :param events: List of events.
        :param to: Destination component.
        """
        return self.queue.publish(events)

    def submit(self, job: Job, dependencies: t.Sequence[str]) -> str:
        """
        Submits a function for local execution.

        :param job: The `Job` to be executed.
        :param dependencies: List of `job_ids`
        """
        component = self._cache[job.type_id, job.identifier, job.uuid]
        method = getattr(component, job.method)
        logging.info(f"Submitting job: {job}")
        method(*job.args, **job.kwargs)
        future_key = str(uuid.uuid4())
        logging.success("Done")
        return future_key

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
