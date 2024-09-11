from collections import defaultdict
import typing as t

from superduper.backends.base.queue import consume_events
from superduper.base.event import Event
from superduper.backends.base.queue import BaseQueueConsumer, BaseQueuePublisher

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
        self._components = {}

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
                self.components[type_id, identifier] = self.db.load(type_id=type_id, identifier=identifier)

    def _put(self, component):
        self.queue[component.type_id, component.identifier] = []
        self._components[component.type_id, component.identifier] = component
        self._component_uuid_mapping[component.type_id, component.identifier] = component.uuid

    def list_components(self):
        return list(self._components.keys())
    
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
            identifier = event.dest.identifier
            type_id = event.dest.type_id
            self.queue[type_id, identifier].append(event)

        return self.consumer.consume(
            db=self.db, queue=self.queue, components=self._components
        )


class LocalQueueConsumer(BaseQueueConsumer):
    """LocalQueueConsumer for consuming message from queue.

    :param uri: Uri to connect.
    :param queue_name: Queue to consume.
    :param callback: Callback for consumed messages.
    """

    def start_consuming(self):
        """Start consuming."""

    def _get_consumers(self, db, components):
        def _remove_duplicates(clist):
            seen = set()
            return [x for x in clist if not (x in seen or seen.add(x))]

        components = list(_remove_duplicates(components.keys()))
        components_to_use = []

        for type_id, _ in components:
            components_to_use += [(type_id, x) for x in db.show(type_id)]

        return set(components_to_use + components)

    def consume(self, db: 'Datalayer', queue: t.Dict, components: t.Dict):
        """Consume the current queue and run jobs."""
        queue_jobs = defaultdict(lambda: [])
        consumers = self._get_consumers(db, components)
        # All consumers are executed one by one.
        for consumer in consumers:
            events = queue[consumer]
            queue[consumer] = []
            component = components[consumer]
            # Consume
            jobs = consume_events(component=component, events=events)
            queue_jobs[consumer].extend(jobs)
        return queue_jobs

    @property
    def db(self):
        """Instance of Datalayer."""
        return self._db

    @db.setter
    def db(self, db):
        self._db = db

    def close_connection(self):
        """Close connection to queue."""
