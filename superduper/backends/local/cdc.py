from superduper.backends.base.cdc import CDCBackend
from superduper.components.cdc import CDC


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

    def _put(self, item):
        assert isinstance(item, CDC)
        self.triggers.add((item.type_id, item.identifier))

    def __delitem__(self, item):
        self.triggers.remove(item)

    def initialize(self):
        """Initialize the CDC."""
        for component_data in self.db.show():
            type_id = component_data['type_id']
            identifier = component_data['identifier']
            r = self.db.show(type_id=type_id, identifier=identifier, version=-1)
            if r.get('trigger'):
                self.put(self.db.load(type_id=type_id, identifier=identifier))
            # TODO consider re-initialzing CDC jobs since potentially failure

    def drop(self):
        """Drop the CDC."""
        self.triggers = set()
        self._trigger_uuid_mapping = {}
