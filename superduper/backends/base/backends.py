import typing as t
from abc import ABC, abstractmethod
from collections import defaultdict

from superduper import logging
from superduper.misc.importing import isreallyinstance

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer
    from superduper.components.component import Component


class Bookkeeping(ABC):
    """Mixin class for tracking components and associated tools."""

    def __init__(self):
        self.component_uuid_mapping = defaultdict(set)
        self.uuid_component_mapping = {}
        self.tool_uuid_mapping = defaultdict(set)
        self.uuid_tool_mapping = {}
        self.tools = {}

    def initialize_with_components(self):
        """Initialize the backend with components.

        This method is executed when a cluster is initialized.
        """
        for info in self.db.show():
            obj = self.db.load(**info)
            if isreallyinstance(obj, self.cls):
                self.put_component(obj)

    def build_tool(self, component: 'Component'):
        """Build a tool from a component.

        :param component: Component to build tool from.
        """
        pass

    def get_tool(self, uuid: str):
        """Get the tool from a uuid.

        :param uuid: UUID of the tool.
        """
        tool_id = self.uuid_tool_mapping[uuid]
        return self.tools[tool_id]

    def put_component(self, component: 'Component', **kwargs):
        """Put a component to the backend.

        :param component: Component to put.
        :param kwargs: kwargs dictionary.
        """
        logging.info(
            f'Putting component: {component.huuid} on to {self.__class__.__name__}'
        )
        tool = self.build_tool(component)
        tool.db = self.db
        self.component_uuid_mapping[(component.component, component.identifier)].add(
            component.uuid
        )
        self.uuid_component_mapping[component.uuid] = (
            component.component,
            component.identifier,
        )
        self.uuid_tool_mapping[component.uuid] = tool.identifier
        if tool.identifier in self.tools:
            return
        self.tool_uuid_mapping[tool.identifier].add(component.uuid)
        self.tools[tool.identifier] = tool
        tool.initialize(**kwargs)

    def drop_component(self, component: str, identifier: str):
        """Drop the component from backend.

        :param component: Component name.
        :param identifier: Component identifier.
        """
        uuids = self.component_uuid_mapping[(component, identifier)]
        tool_ids = []
        for uuid in uuids:
            del self.uuid_component_mapping[uuid]
            tool_id = self.uuid_tool_mapping[uuid]
            tool_ids.append(tool_id)
            del self.uuid_tool_mapping[uuid]
            self.tool_uuid_mapping[tool_id].remove(uuid)
            if not self.tool_uuid_mapping[tool_id]:
                self.tools[tool_id].drop()
                del self.tools[tool_id]
        del self.component_uuid_mapping[(component, identifier)]

    def drop(self):
        """Drop the backend."""
        for tool in self.tools.values():
            tool.drop()
        self.component_uuid_mapping = defaultdict(set)
        self.uuid_component_mapping = {}
        self.tool_uuid_mapping = defaultdict(set)
        self.uuid_tool_mapping = {}
        self.tools = {}

    def list_components(self):
        """List components, and identifiers deployed."""
        return list(self.component_uuid_mapping.keys())

    def list_tools(self):
        """List tools deployed."""
        return list(self.tools.keys())

    def list_uuids(self):
        """List uuids deployed."""
        return list(self.uuid_component_mapping.keys())


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
        """List components, and identifiers deployed."""
        pass

    @abstractmethod
    def initialize(self):
        """To be called on program start."""

    @abstractmethod
    def put_component(self, component: 'Component'):
        """Add a component to the deployment.

        :param component: ``Component`` to put.
        """

    @abstractmethod
    def drop_component(self, component: str, identifier: str):
        """Drop the component from backend.

        :param component: Component name.
        :param identifier: Component identifier.
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
