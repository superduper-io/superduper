from collections import defaultdict
import typing as t

from superduper.backends.base.compute import ComputeBackend
from superduper.base.event import Job


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
        self.futures = defaultdict(lambda: {})

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

    def release_futures(self, context: str):
        try:
            del self.futures[context]
        except KeyError:
            pass

    def component_hook(self, *args, **kwargs):
        """Hook for component."""
        pass

    def submit(self, job: Job) -> str:
        """
        Submits a function for local execution.

        :param job: The `Job` to be executed.
        :param dependencies: List of `job_ids`
        """
        output = job(
            db=self.db,
            futures=self.futures[job.context],
        )
        self.futures[job.context][job.job_id] = output

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
