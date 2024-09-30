import typing as t

from superduper.backends.base.queue import (
    BaseQueueConsumer,
    BaseQueuePublisher,
    consume_streaming_events,
)
from superduper.base.event import Event
from superduper.components.cdc import CDC

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


class LocalQueuePublisher(BaseQueuePublisher):
    """
    LocalQueuePublisher for handling publisher and consumer process.

    Local queue which holds listeners, vector indices as queue which
    consists of events to be consumed by the corresponding components.

    :param uri: uri to connect.
    """

    def __init__(self, uri: t.Optional[str] = None):
        super().__init__(uri=uri)
        self.consumer = self.build_consumer()
        self._component_uuid_mapping = {}

    def list(self):
        """List all components."""
        return self.queues.keys()

    def drop(self):
        self.queues = {}

    def __delitem__(self, item):
        del self.queues[item]

    def initialize(self):
        for type_id, identifier in self.db.show():
            r = self.db.show(type_id=type_id, identifier=identifier, version=-1)
            if r['trigger']:
                self.queue[type_id, identifier] = []

    def _put(self, component):
        msg = 'Table name "_apply" collides with Superduper namespace'
        assert component.cdc_table != '_apply', msg
        assert isinstance(component, CDC)
        self._component_uuid_mapping[
            component.type_id, component.identifier
        ] = component.uuid
        if component.cdc_table in self.queue:
            return
        self.queue[component.cdc_table] = []

    def list_components(self):
        return list(self._component_uuid_mapping.keys())

    def list_uuids(self):
        return list(self._component_uuid_mapping.values())

    def build_consumer(self, **kwargs):
        """Build consumer client."""
        return LocalQueueConsumer()

    def publish(self, events: t.List[Event]):
        """
        Publish events to local queue.

        :param events: list of events
        """
        for event in events:
            self.queue[event.queue].append(event)
        self.consumer.consume(db=self.db, queue=self.queue)


class LocalQueueConsumer(BaseQueueConsumer):
    """LocalQueueConsumer for consuming message from queue.

    :param uri: Uri to connect.
    :param queue_name: Queue to consume.
    :param callback: Callback for consumed messages.
    """

    def start_consuming(self):
        """Start consuming."""

    def consume(self, db: 'Datalayer', queue: t.Dict[str, Event]):
        """Consume the current queue and run jobs."""
        keys = list(queue.keys())[:]
        for k in keys:
            if k != '_apply':
                consume_streaming_events(events=queue[k], table=k, db=db)
                queue[k] = []
            else:
                while queue['_apply']:
                    event = queue['_apply'].pop(0)
                    event.execute(db)

    @property
    def db(self):
        """Instance of Datalayer."""
        return self._db

    @db.setter
    def db(self, db):
        self._db = db

    def close_connection(self):
        """Close connection to queue."""
