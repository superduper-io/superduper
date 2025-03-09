from collections import defaultdict
import typing as t
from abc import ABC, abstractmethod

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer
    from superduper.components.component import Component


class Bookkeeping(ABC):
    def __init__(self):
        self.component_uuid_mapping = defaultdict(set)
        self.uuid_component_mapping = {}
        self.tool_uuid_mapping = defaultdict(set)
        self.uuid_tool_mapping = {}
        self.tools = {}

    def build_tool(self, component: 'Component'):
        pass

    def get_tool(self, uuid: str):
        tool_id = self.uuid_tool_mapping[uuid]
        return self.tools[tool_id]

    def put_component(self, component: 'Component', **kwargs):
        tool = self.build_tool(component)
        tool.db = self.db
        self.component_uuid_mapping[(component.component, component.identifier)].add(component.uuid)
        self.uuid_component_mapping[component.uuid] = (component.component, component.identifier)
        self.uuid_tool_mapping[component.uuid] = tool.identifier
        self.tool_uuid_mapping[tool.identifier].add(component.uuid)
        self.tools[tool.identifier] = tool
        tool.initialize(**kwargs)

    def drop_component(self, component: str, identifier: str):
        uuids = self.component_uuid_mapping[(component, identifier)]
        tool_ids = []
        for uuid in uuids:
            del self.uuid_component_mapping[uuid]
            tool_id = self.uuid_tool_mapping[uuid]
            tool_ids.append(tool_id)
            del self.uuid_tool_mapping[uuid]
            self.tool_uuid_mapping[tool_id].remove(uuid)
            if not self.tool_uuid_mapping[tool_id]:
                self.tools.drop()
                del self.tools[tool_id]
        del self.component_uuid_mapping[(component, identifier)]

    def drop(self):
        for tool in self.tools.values():
            tool.drop()
        self.component_uuid_mapping = defaultdict(set)
        self.uuid_component_mapping = {}
        self.tool_uuid_mapping = defaultdict(set)
        self.uuid_tool_mapping = {}
        self.tools = {}

    def list_components(self):
        return list(self.component_uuid_mapping.keys())

    def list_tools(self):
        return list(self.tools.keys())

    def list_uuids(self):
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
        pass

    @abstractmethod
    def put_component(self, component: 'Component'):
        """Add a component to the deployment.

        :param component: ``Component`` to put.
        :param kwargs: kwargs dictionary.
        """

    @abstractmethod
    def drop_component(self, component: str, identifier: str):
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
