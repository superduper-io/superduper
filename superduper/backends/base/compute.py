import typing as t
from abc import abstractmethod

from superduper.backends.base.backends import BaseBackend
from superduper.base.event import Job

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


class ComputeBackend(BaseBackend):
    """
    Abstraction for sending jobs to a distributed compute platform.

    :param args: *args for `ABC`
    :param kwargs: *kwargs for `ABC`

    # noqa
    """

    @property
    @abstractmethod
    def type(self) -> str:
        """Return the type of compute engine."""
        pass

    # TODO is this used anywhere?
    @abstractmethod
    def release_futures(self, context: str):
        """Release futures from backend."""
        pass

    @property
    @abstractmethod
    def remote(self) -> bool:
        """Return if remote compute engine."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of current compute engine."""
        pass

    def get_local_client(self):
        """Returns a local version of self."""
        pass

    @abstractmethod
    def submit(self, job: Job) -> t.Any:
        """
        Submits a function to the server for execution.

        :param job: The ``Job`` to be executed.
        """
        pass

    # TODO is this used?
    @property
    @abstractmethod
    def tasks(self) -> t.Any:
        """List for all tasks."""
        pass

    # TODO is this used?
    @abstractmethod
    def wait_all(self) -> None:
        """Waits for all pending tasks to complete."""
        pass

    # TODO is this used?
    @abstractmethod
    def result(self, identifier: str) -> t.Any:
        """Retrieves the result of a previously submitted task.

        Note: This will block until the future is completed.

        :param identifier: The identifier of the submitted task.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect the client."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shuts down the compute cluster."""
        pass

    def execute_task(self, job_id, dependencies, compute_kwargs={}):
        """Execute task function for distributed backends."""

    def initialize(self):
        """Connect to address."""

    def create_handler(self, *args, **kwargs):
        """Create handler on component declare."""

    @property
    def db(self) -> 'Datalayer':
        """Get the ``db``."""
        return self._db

    @db.setter
    def db(self, value: 'Datalayer'):
        """Set the ``db``.

        :param value: ``Datalayer`` instance.
        """
        self._db = value
