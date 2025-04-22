from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple

from superduper.backends.base.cdc import CDCBackend
from superduper.base.datalayer import Datalayer
from superduper.components.cdc import CDC

if TYPE_CHECKING:
    from superduper import Component


class LocalCDCBackend(CDCBackend):
    """Local CDC backend."""

    def __init__(self, db: Datalayer):
        super().__init__()

        assert db, "Empty datalayer"
        self._db = db

        # Based on put_component usage: (component.component, component.identifier)
        self.triggers: Set[Tuple[str, str]] = set()
        # Currently not used in the code but assuming it maps some key to UUID strings
        self._trigger_uuid_mapping: Dict[str, str] = {}

    def handle_event(self, table, ids, event_type):
        """Handle an event.

        :param table: The table.
        :param ids: The IDs.
        :param event_type: The event type.
        """
        return self._db.on_event(table=table, ids=ids, event_type=event_type)

    def list_components(self):
        """List components."""
        return sorted(list(self.triggers))

    def list_uuids(self):
        """List UUIDs of components."""
        return list(self._trigger_uuid_mapping.values())

    def put_component(self, component):
        assert isinstance(component, CDC)
        self.triggers.add((component.component, component.identifier))

    def drop_component(self, component, identifier):
        c = self._db.load(component=component, identifier=identifier)
        if isinstance(c, CDC):
            # Note: This looks like a potential bug - it's trying to remove 'cdc_table'
            # but 'triggers' contains tuples of (component, identifier)
            self.triggers.remove(c.cdc_table)  # This might need to be fixed

    def initialize(self):
        """Initialize the CDC."""
        for component_data in self._db.show():
            component = component_data['component']
            identifier = component_data['identifier']
            r = self._db.show(component=component, identifier=identifier, version=-1)
            if r.get('trigger'):
                self.put_component(
                    self._db.load(component=component, identifier=identifier)
                )
            # TODO consider re-initialzing CDC jobs since potentially failure

    def drop(self, component: Optional['Component'] = None):
        """Drop the CDC.

        :param component: Component to remove.
        """
        self.triggers = set()
        self._trigger_uuid_mapping = {}
