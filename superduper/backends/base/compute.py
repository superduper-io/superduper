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

    @abstractmethod
    def release_futures(self, context: str):
        """Release futures from backend.

        :param context: Context of futures to release.
        """
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

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect the client."""
        pass

    def initialize(self):
        """Connect to address."""

    def create_handler(self, *args, **kwargs):
        """Create handler on component declare.

        :param args: *args for `create_handler`
        :param kwargs: *kwargs for `create_handler`
        """

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

    def drop_component(self, uuid: str):
        """Drop the component from compute.

        :param uuid: Component uuid.
        """
