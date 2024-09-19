import typing as t
import uuid

from superduper import logging
from superduper.backends.base.compute import ComputeBackend
from superduper.jobs.job import Job


class LocalComputeBackend(ComputeBackend):
    """
    A mockup backend for running jobs locally.

    :param uri: Optional uri param.
    :param kwargs: Optional kwargs.
    """

    def __init__(
        self,
        uri: t.Optional[str] = None,
        **kwargs,
    ):
        self.uri = uri
        self.kwargs = kwargs
        self._cache: t.Dict = {}
        self._db = None

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

    def submit(self, job: Job, dependencies: t.Sequence[str]) -> str:
        """
        Submits a function for local execution.

        :param job: The `Job` to be executed.
        :param dependencies: List of `job_ids`
        """
        component = self.db.load(uuid=job.uuid)
        method = getattr(component, job.method)
        logging.info(f"Submitting job: {job}")
        # TODO for distributed computation this is a future
        # which gets returned
        return method(*job.args, **job.kwargs)

    def __delitem__(self, item):
        pass

    def _put(self, component):
        """Deploy a component on the compute."""
        pass

    def list_components(self):
        """List all components on the compute."""
        return []

    def list_uuids(self):
        """List all UUIDs on the compute."""
        return []

    def initialize(self):
        pass

    def drop(self):
        pass

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
