import typing as t
from abc import ABC, abstractmethod
from collections import defaultdict

from superduper import logging
from superduper.base.event import Event

DependencyType = t.Union[t.Dict[str, str], t.Sequence[t.Dict[str, str]]]

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


class BaseQueueConsumer(ABC):
    """
    Base class for handling consumer process.

    This class is an implementation of message broker between
    producers (superduper db client) and consumers i.e listeners.

    :param uri: Uri to connect.
    :param queue_name: Queue to consume.
    :param callback: Callback for consumed messages.
    """

    def __init__(
        self,
        uri: t.Optional[str] = '',
        queue_name: str = '',
        callback: t.Optional[t.Callable] = None,
    ):
        self.uri = uri
        self.callback = callback
        self.queue_name = queue_name

    @abstractmethod
    def start_consuming(self):
        """Abstract method to start consuming messages."""
        pass

    @abstractmethod
    def close_connection(self):
        """Abstract method to close connection."""
        pass

    def consume(self, *args, **kwargs):
        """Start consuming messages from queue."""
        logging.info(f"Started consuming on queue: {self.queue_name}")
        try:
            self.start_consuming()
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt: Stopping consumer...")
        finally:
            self.close_connection()
            logging.info(f"Stopped consuming on queue: {self.queue_name}")


class BaseQueuePublisher(ABC):
    """
    Base class for handling publisher and consumer process.

    This class is an implementation of message broker between
    producers (superduper db client) and consumers i.e listeners.

    :param uri: Uri to connect.
    """

    def __init__(self, uri: t.Optional[str]):
        self.uri: t.Optional[str] = uri
        self.queue: t.Dict = defaultdict(lambda: [])

        from superduper.misc.special_dicts import ArgumentDefaultDict

        self.components = ArgumentDefaultDict(
            default_factory=lambda x: self.db.load(*x)
        )
        self._db = None
        self._component_map: t.Dict = {}

    @property
    def db(self):
        """Instance of Datalayer."""
        return self._db

    @db.setter
    def db(self, db):
        self._db = db

    @abstractmethod
    def build_consumer(self, **kwargs):
        """Build a consumer instance."""

    @abstractmethod
    def publish(self, events: t.List[Event]):
        """
        Publish events to local queue.

        :param events: list of events
        :param to: Component name for events to be published.
        """

    @abstractmethod
    def declare_component(self, component):
        """Declare component and add it to queue."""


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

    def build_consumer(self, **kwargs):
        """Build consumer client."""
        return LocalQueueConsumer()

    def declare_component(self, component):
        """Declare component and add it to queue."""
        self.components[component.type_id, component.identifier] = component

    def publish(self, events: t.List[Event]):
        """
        Publish events to local queue.

        :param events: list of events
        """
        for event in events:
            identifier = event.identifier
            type_id = event.type_id
            self._component_map.update({'identifier': identifier, 'type_id': type_id})
            self.queue[type_id, identifier].append(event)

        return self.consumer.consume(
            db=self.db, queue=self.queue, components=self.components
        )


class LocalQueueConsumer(BaseQueueConsumer):
    """LocalQueueConsumer for consuming message from queue.

    :param uri: Uri to connect.
    :param queue_name: Queue to consume.
    :param callback: Callback for consumed messages.
    """

    def start_consuming(self):
        """Start consuming."""

    def consume(self, db: 'Datalayer', queue: t.Dict, components: t.Dict):
        """Consume the current queue and run jobs."""
        from superduper.base.datalayer import DBEvent

        queue_jobs = defaultdict(lambda: [])
        components_to_use = [('listener', x) for x in db.show('listener')]
        components_to_use += [('vector_index', x) for x in db.show('vector_index')]
        for type_id, identifier in components_to_use:
            events = queue[type_id, identifier]
            if not events:
                continue
            queue[type_id, identifier] = []
            component = components[type_id, identifier]
            jobs = []
            for event_type, type_events in DBEvent.chunk_by_event(events).items():
                ids = [event.id for event in type_events]
                overwrite = (
                    True if event_type in [DBEvent.insert, DBEvent.upsert] else False
                )
                logging.info(
                    f'Running jobs for {component.type_id}::{component.identifier}'
                )
                logging.debug(f'Using ids: {ids}')
                job = component.run_jobs(
                    db=db, ids=ids, overwrite=overwrite, event_type=event_type
                )
                jobs.append(job)
            queue_jobs[type_id, identifier].extend(jobs)
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
