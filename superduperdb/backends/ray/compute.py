import json
import os
import typing as t
import uuid

import ray
from ray.job_submission import JobSubmissionClient

from superduperdb import logging
from superduperdb.backends.base.compute import ComputeBackend


class RayComputeBackend(ComputeBackend):
    """A client for interacting with a ray cluster. Initialize the ray client.

    :param address: The address of the ray cluster.
    :param local: Set to True to create a local Dask cluster. (optional)
    :param kwargs: Additional keyword arguments to be passed to the ray client.
    """

    def __init__(
        self,
        address: t.Optional[str] = None,
        local: bool = False,
        **kwargs,
    ):
        self._futures_collection: t.Dict[str, ray.ObjectRef] = {}
        self.address = address

        self.client = JobSubmissionClient(self.address)

    @property
    def remote(self) -> bool:
        """Return if remote compute engine."""
        return True

    @property
    def type(self) -> str:
        """The type of the compute backend."""
        return "distributed"

    @property
    def name(self) -> str:
        """The name of the compute backend."""
        return f"ray://{self.address}"

    def submit(self, identifier, dependencies=(), compute_kwargs={}):
        """
        Submit job to remote cluster.

        :param identifier: Job identifier.
        :param dependencies: List of dependencies on the job.
        :param compute_kwargs: Compute kwargs for the job.
        """
        try:
            uuid.UUID(str(identifier))
        except ValueError:
            raise ValueError(f'Identifier {identifier} is not valid')
        dependencies = list([d for d in dependencies if d is not None])

        if dependencies:
            dependencies = f"dependencies={json.dumps(dependencies)}"
            job_string = f"remote_job(\"{identifier}\", {dependencies}"
        else:
            job_string = f"remote_job(\"{identifier}\""

        if compute_kwargs:
            job_string += f", compute_kwargs={json.dumps(compute_kwargs)})"
        else:
            job_string += ")"

        entrypoint = (
            f"python -c 'from superduperdb.jobs.job import remote_job; {job_string}'"
        )

        runtime_env = {}
        env_vars = {
            k: os.environ[k] for k in os.environ if k.startswith('SUPERDUPERDB_')
        }
        if env_vars:
            runtime_env = {'env_vars': env_vars}
        job_id = self.client.submit_job(entrypoint=entrypoint, runtime_env=runtime_env)
        return job_id

    def execute_task(
        self, job_id, dependencies, compute_kwargs={}
    ) -> t.Tuple[ray.ObjectRef, str]:
        """
        Submits a function to the ray server for execution.

        :param function: The function to be executed.
        :param args: Positional arguments to be passed to the function.
        :param kwargs: Keyword arguments to be passed to the function.
        """

        def _dependable_remote_job(function, *args, **kwargs):
            if 'dependencies' in kwargs:
                dependencies = kwargs.pop('dependencies', None)
                if dependencies:
                    ray.wait(dependencies)
            return function(*args, **kwargs)

        if compute_kwargs:
            remote_function = ray.remote(**compute_kwargs)(_dependable_remote_job)
        else:
            remote_function = ray.remote(_dependable_remote_job)

        from superduperdb.jobs.job import remote_task

        future = remote_function.remote(remote_task, job_id, dependencies=dependencies)

        ray.get(future)
        task_id = str(future.task_id().hex())
        self._futures_collection[task_id] = future

        logging.success(
            f"Job submitted on {self}.  function: remote_job; "
            f"task: {task_id}; job_id: {str(future.job_id())}"
        )
        return future, task_id

    @property
    def tasks(self) -> t.Dict[str, ray.ObjectRef]:
        """List all pending tasks."""
        return self._futures_collection

    def wait(self, identifier: str) -> None:
        """Waits for task corresponding to identifier to complete.

        :param identifier: Future task id to wait
        """
        ray.wait([self._futures_collection[identifier]])

    def wait_all(self) -> None:
        """Waits for all tasks to complete."""
        ray.wait(
            list(self._futures_collection.values()),
            num_returns=len(self._futures_collection),
        )

    def result(self, identifier: str) -> t.Any:
        """Retrieves the result of a previously submitted task.

        Note: This will block until the future is completed.

        :param identifier: The identifier of the submitted task.
        """
        future = self._futures_collection[identifier]
        return ray.get(future)

    def disconnect(self) -> None:
        """Disconnect the ray client."""
        ray.shutdown()

    def shutdown(self) -> None:
        """Shuts down the ray cluster."""
        raise NotImplementedError
