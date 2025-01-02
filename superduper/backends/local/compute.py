import typing as t
from collections import defaultdict

from superduper import logging
from superduper.backends.base.compute import ComputeBackend
from superduper.base.event import Job

if t.TYPE_CHECKING:
    from superduper import Component


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
        self.futures: t.DefaultDict = defaultdict(lambda: {})

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

    # TODO needed?
    def release_futures(self, context: str):
        """Release futures for a given context."""
        try:
            del self.futures[context]
        except KeyError:
            logging.warn(f'Could not release futures for context {context}')

    # TODO needed? (we have .put)
    # TODO hook to do what?
    def component_hook(self, *args, **kwargs):
        """Hook for component."""
        pass

    def submit(self, job: Job) -> str:
        """
        Submits a function for local execution.

        :param job: The `Job` to be executed.
        :param dependencies: List of `job_ids`
        """
        args, kwargs = job.get_args_kwargs(self.futures[job.context])

        assert job.job_id is not None
        component = self.db.load(uuid=job.uuid)
        self.db.metadata.update_job(job.job_id, 'status', 'running')

        try:
            logging.debug(
                f'Running job {job.job_id}: {component.identifier}.{job.method}'
            )
            method = getattr(component, job.method)
            output = method(*args, **kwargs)
        except Exception as e:
            self.db.metadata.update_job(job.job_id, 'status', 'failed')
            raise e

        self.db.metadata.update_job(job.job_id, 'status', 'success')
        self.futures[job.context][job.job_id] = output
        assert job.job_id is not None
        return job.job_id

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
        """Initialize the compute."""
        pass

    def drop(self, component: t.Optional['Component'] = None):
        """Drop the compute.

        :param component: Component to remove.
        """

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
