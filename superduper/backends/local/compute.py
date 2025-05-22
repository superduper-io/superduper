import typing as t
from collections import defaultdict

from superduper import logging
from superduper.backends.base.compute import ComputeBackend
from superduper.base.metadata import STATUS_FAILED, Job
from superduper.base.status import STATUS_RUNNING, STATUS_SUCCESS

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

    def release_futures(self, context: str):
        """Release futures for a given context.

        :param context: The apply context to release futures for.
        """
        try:
            del self.futures[context]
        except KeyError:
            logging.warn(f'Could not release futures for context {context}')

    def submit(self, job: Job) -> str:
        """
        Submits a function for local execution.

        :param job: The `Job` to be executed.
        """
        job.args, job.kwargs = job.get_args_kwargs(self.futures[job.context])
        dependencies = job.kwargs.pop('dependencies', [])
        if dependencies:
            logging.info(
                f'Running job {job.job_id} with {len(dependencies)} dependencies'
            )
        output = job.run(db=self.db)
        self.futures[job.context][job.job_id] = output
        assert job.job_id is not None
        return job.job_id

    def list_components(self):
        """List all components on the compute."""
        return []

    def list_uuids(self):
        """List all UUIDs on the compute."""
        return []

    def drop_component(self, component: str, identifier: str):
        """Drop a component from the compute."""

    def put_component(self, component: str, uuid: str):
        """Create a handler on compute."""

    def initialize(self):
        """Initialize the compute."""

    def remote(self, uuid: str, component: str, method: str, *args, **kwargs):
        """Run a remote method on the compute."""
        raise NotImplementedError

    def drop(self):
        """Drop the compute.

        :param component: Component to remove.
        """

    def disconnect(self) -> None:
        """Disconnect the local client."""
