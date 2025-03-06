import threading
import typing as t

from superduper import logging
from superduper.backends.base.compute import ComputeBackend
from superduper.backends.base.scheduler import (
    BaseQueueConsumer,
    BaseScheduler,
    consume_events,
)
from superduper.base.event import Event
from superduper.components.cdc import CDC

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


class LocalScheduler(BaseScheduler):
    """
    Class for handling publisher and consumer processes.

    Contains a local queue which holds listeners, vector indices in a queue which
    consists of events to be consumed by the corresponding components.

    :param uri: uri to connect.
    """

    def __init__(self, compute: ComputeBackend):
        super().__init__()
        self.compute = compute
        self.consumer = LocalQueueConsumer()
        self._component_uuid_mapping: t.Dict = {}
        self.lock = threading.Lock()

    @property
    def db(self):
        return self._db

    @db.setter
    def db(self, value):
        self._db = value
        self.compute.db = value

    def drop(self):
        """Drop the queue."""
        self.queue = {}

    def drop_component(self, component, identifier):
        c = self.db.load(component=component, identifier=identifier)
        if isinstance(c, CDC):
            del self.queue[c.cdc_table]

    def initialize(self):
        """Initialize the queue."""
        for component_data in self.db.show():
            component = component_data['component']
            identifier = component_data['identifier']
            r = self.db.show(component=component, identifier=identifier, version=-1)
            if r.get('trigger'):
                with self.lock:
                    self.queue[component, identifier] = []

    def _put(self, component):
        msg = 'Table name "_apply" collides with Superduper namespace'
        assert component.cdc_table != '_apply', msg
        assert isinstance(component, CDC)
        self._component_uuid_mapping[component.component, component.identifier] = (
            component.uuid
        )
        if component.cdc_table in self.queue:
            return
        self.queue[component.cdc_table] = []

    def list_components(self):
        """List all components."""
        return list(self._component_uuid_mapping.keys())

    def list_uuids(self):
        """List all UUIDs."""
        return list(self._component_uuid_mapping.values())

    def publish(self, events: t.List[Event]):
        """
        Publish events to local queue.

        :param events: list of events
        """
        with self.lock:
            for event in events:
                self.queue[event.queue].append(event)
            self.consumer.consume(db=self.db, compute=self.compute, queue=self.queue)


class LocalQueueConsumer(BaseQueueConsumer):
    """LocalQueueConsumer for consuming message from queue.

    :param uri: Uri to connect.
    :param queue_name: Queue to consume.
    :param callback: Callback for consumed messages.
    """

    def start_consuming(self):
        """Start consuming."""

    def consume(self, db: 'Datalayer', compute: ComputeBackend, queue: t.Dict[str, t.List[Event]]):
        """Consume the current queue and run jobs.

        :param db: Datalayer instance.
        :param queue: Queue to consume.
        """
        keys = list(queue.keys())[:]
        for k in keys:
            consume_events(events=queue[k], table=k, db=db, compute=compute)
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
