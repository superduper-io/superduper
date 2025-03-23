import typing as t
from abc import abstractmethod

from superduper.backends.base.backends import BaseBackend
from superduper.misc.importing import isreallysubclass

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer
    from superduper.base.event import Job
    from superduper.components.component import Component


class ComputeBackend(BaseBackend):
    """
    Abstraction for sending jobs to a distributed compute platform.

    :param args: *args for `ABC`
    :param kwargs: *kwargs for `ABC`

    # noqa
    """

    @abstractmethod
    def release_futures(self, context: str):
        """Release futures from backend.

        :param context: Futures context to release.
        """
        pass

    @abstractmethod
    def submit(self, job: 'Job') -> t.Any:
        """
        Submits a function to the server for execution.

        :param job: The ``Job`` to be executed.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect the client."""
        pass

    @abstractmethod
    def remote(self, uuid: str, function: str, *args, **kwargs):
        """
        Call a remote function on the server.

        :param uuid: The UUID of the component.
        :param function: The function to call.
        :param args: The arguments to pass to the function.
        :param kwargs: The keyword arguments to pass to the function.
        """
        pass

    def initialize(self):
        """Connect to address."""
        from superduper.components.model import Model

        metadata = self.db.load('Table')
        for c in metadata:
            if isreallysubclass(c.cls, Model) and c.serve:
                for m in self.db.load(c.identifier):
                    self.put_component(m)

    @abstractmethod
    def put_component(self, component: 'Component'):
        """Create handler on component declare.

        :param component: Component to put.
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

    @abstractmethod
    def drop_component(self, component: str, identifier: str):
        """Drop the component from compute.

        :param component: Component name.
        :param identifier: Component identifier.
        """
