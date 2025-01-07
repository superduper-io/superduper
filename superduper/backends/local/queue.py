import threading
import typing as t

from superduper import logging
from superduper.backends.base.queue import (
    BaseQueueConsumer,
    BaseQueuePublisher,
    consume_events,
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
        self._component_uuid_mapping: t.Dict = {}
        self.lock = threading.Lock()

    def show_pending_create_events(self, type_id: str | None = None):
        if type_id is None:
            return [
                {'type_id': type_id, 'identifier': e.component['identifier']}
                for e in self.queue['_apply']
            ]
        else:
            return [
                e.component['identifier']
                for e in self.queue['_apply']
                if e.component['type_id'] == type_id
            ]

    def list(self):
        """List all components."""
        return self.queue.keys()

    def drop(self):
        """Drop the queue."""
        self.queue = {}

    def __delitem__(self, item):
        del self.queue[item]

    def initialize(self):
        """Initialize the queue."""
        for component_data in self.db.show():
            type_id = component_data['type_id']
            identifier = component_data['identifier']
            r = self.db.show(type_id=type_id, identifier=identifier, version=-1)
            if r.get('trigger'):
                with self.lock:
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
        """List all components."""
        return list(self._component_uuid_mapping.keys())

    def list_uuids(self):
        """List all UUIDs."""
        return list(self._component_uuid_mapping.values())

    def build_consumer(self, **kwargs):
        """Build consumer client."""
        return LocalQueueConsumer()

    def publish(self, events: t.List[Event]):
        """
        Publish events to local queue.

        :param events: list of events
        """
        with self.lock:
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

    def consume(self, db: 'Datalayer', queue: t.Dict[str, t.List[Event]]):
        """Consume the current queue and run jobs."""
        keys = list(queue.keys())[:]
        for k in keys:
            consume_events(events=queue[k], table=k, db=db)
            queue[k] = []

        logging.info('Consumed all events')

    @property
    def db(self):
        """Instance of Datalayer."""
        return self._db

    @db.setter
    def db(self, db):
        self._db = db

    def close_connection(self):
        """Close connection to queue."""
