import typing as t

from superduper.backends.base.cdc import CDCBackend
from superduper.components.cdc import CDC

if t.TYPE_CHECKING:
    from superduper import Component


class LocalCDCBackend(CDCBackend):
    """Local CDC backend."""

    def __init__(self):
        super().__init__()
        self.triggers = set()
        self._trigger_uuid_mapping = {}

    def handle_event(self, table, ids, event_type):
        """Handle an event.

        :param table: The table.
        :param ids: The IDs.
        :param event_type: The event type.
        """
        return self.db.on_event(table=table, ids=ids, event_type=event_type)

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
        c = self.db.load(component=component, identifier=identifier)
        if isinstance(c, CDC):
            self.triggers.remove(c.cdc_table)

    def initialize(self):
        """Initialize the CDC."""
        for component_data in self.db.show():
            component = component_data['component']
            identifier = component_data['identifier']
            r = self.db.show(component=component, identifier=identifier, version=-1)
            if r.get('trigger'):
                self.put_component(
                    self.db.load(component=component, identifier=identifier)
                )
            # TODO consider re-initialzing CDC jobs since potentially failure

    def drop(self, component: t.Optional['Component'] = None):
        """Drop the CDC.

        :param component: Component to remove.
        """
        self.triggers = set()
        self._trigger_uuid_mapping = {}
