# TODO deprecate the implementations in favour of the plugin paradigm
import typing as t
from abc import ABC, abstractmethod

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer
    from superduper.components.component import Component


class BaseBackend(ABC):
    """Base backend class for cluster client."""

    def __init__(self):
        self._db = None

    @abstractmethod
    def drop(self, component: t.Optional['Component'] = None):
        """Drop the backend.

        :param component: Component to Drop.
        """

    @abstractmethod
    def list_uuids(self):
        """List uuids deployed."""
        pass

    @abstractmethod
    def list_components(self):
        """List type_ids, and identifiers deployed."""
        pass

    @abstractmethod
    def _put(self, item):
        pass

    @abstractmethod
    def __delitem__(self, item):
        pass

    @abstractmethod
    def initialize(self):
        """To be called on program start."""
        pass

    def put(self, component: 'Component', **kwargs):
        """Add a component to the deployment.

        :param component: ``Component`` to put.
        :param kwargs: kwargs dictionary.
        """
        # This is to make sure that we only have 1 version
        # of each component implemented at any given time
        # TODO: get identifier in string component argument.
        identifier = ''
        if isinstance(component, str):
            uuid = component
        else:
            uuid = component.uuid
            identifier = component.identifier

        if uuid in self.list_uuids():
            return
        if identifier in self.list_components():
            del self[component.identifier]
        self._put(component, **kwargs)

    def drop_component(self, identifier: str):
        """Drop the component from backend.

        :param identifier: Component identifier
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
