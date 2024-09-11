from superduper.backends.base.cdc import CDCBackend
from superduper.components.trigger import Trigger


class LocalCDCBackend(CDCBackend):
    """Local CDC backend."""
    def __init__(self):
        super().__init__()
        self.triggers = set()
        self._trigger_uuid_mapping = {}

    def handle_event(self, query, ids, event_type):
        return self.db.on_event(query=query, ids=ids, event_type=event_type)

    def list_components(self):
        return sorted(list(self.triggers))

    def list_uuids(self):
        return list(self._trigger_uuid_mapping.values())

    def _put(self, item):
        assert isinstance(item, Trigger)
        self.triggers.add((item.type_id, item.identifier))

    def __delitem__(self, item):
        self.triggers.remove(item)

    def initialize(self):
        for type_id, identifier in self.db.show():
            r = self.db.show(type_id=type_id, identifier=identifier, version=-1)
            if r['trigger']:
                self.put(self.db.load(type_id=type_id, identifier=identifier))
            # TODO consider re-initialzing CDC jobs since potentially failure

    def drop(self):
        self.triggers = set()
        self._trigger_uuid_mapping = {}