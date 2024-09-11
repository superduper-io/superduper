import typing as t
from abc import ABC, abstractmethod
from collections import defaultdict

from superduper import logging
from superduper.backends.base.backends import BaseBackend
from superduper.base.event import Event
from superduper.components.component import Component

DependencyType = t.Union[t.Dict[str, str], t.Sequence[t.Dict[str, str]]]


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


class BaseQueuePublisher(BaseBackend):
    """
    Base class for handling publisher and consumer process.

    This class is an implementation of message broker between
    producers (superduper db client) and consumers i.e listeners.

    :param uri: Uri to connect.
    """

    def __init__(self, uri: t.Optional[str]):
        super().__init__()
        self.uri: t.Optional[str] = uri
        self.queue: t.Dict = defaultdict(lambda: [])

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


def consume_events(component: 'Component', events: t.Sequence[Event]):
    """Consume events from queue.

    :param component: Superduper component.
    :param events: Events to be consumed.
    """
    if not events:
        return []
    # Why do we need to chunk events by type?
    event_lookup = Event.chunk_by_type(events)
    for type in event_lookup:
        logging.info(f"Running jobs for {type}")
        component.run_jobs(event=event_lookup[type])
    return []