import threading
import typing as t

from superduper import logging
from superduper.backends.base.backends import Bookkeeping
from superduper.backends.base.compute import ComputeBackend
from superduper.backends.base.scheduler import (
    BaseScheduler,
    consume_events,
)
from superduper.base.event import Event
from superduper.components.cdc import CDC


class QueueWrapper:
    def __init__(self, queue_name: str, scheduler: 'LocalScheduler'):
        self.scheduler = scheduler
        self.queue_name = queue_name

    def initialize(self):
        self.scheduler.Q[self.queue_name] = []

    def drop(self):
        del self.scheduler.Q[self.queue_name]


class LocalScheduler(Bookkeeping, BaseScheduler):
    """
    Class for handling publisher and consumer processes.

    Contains a local queue which holds listeners, vector indices in a queue which
    consists of events to be consumed by the corresponding components.

    :param uri: uri to connect.
    """

    def __init__(self, compute: ComputeBackend):
        self.compute = compute
        self.lock = threading.Lock()
        self.Q = {}

    @property
    def db(self):
        return self._db

    @db.setter
    def db(self, value):
        self._db = value
        self.compute.db = value

    def drop(self):
        """Drop the queue."""
        self.Q = {}

    def build_tool(self, component, uuid):
        c = self.db.load(component=component, uuid=uuid)
        return super().build_tool(component, uuid)

    def drop_component(self, component, identifier):
        c = self.db.load(component=component, identifier=identifier)
        if isinstance(c, CDC):
            del self.Q[c.cdc_table]

    def initialize(self):
        """Initialize the queue."""
        for component_data in self.db.show():
            component = component_data['component']
            identifier = component_data['identifier']
            r = self.db.show(component=component, identifier=identifier, version=-1)
            if r.get('trigger'):
                with self.lock:
                    self.Q[component, identifier] = []

    def put_component(self, component):
        msg = 'Table name "_apply" collides with Superduper namespace'
        assert component.cdc_table != '_apply', msg
        assert isinstance(component, CDC)
        self._component_uuid_mapping[component.component, component.identifier] = (
            component.uuid
        )
        if component.cdc_table in self.Q:
            return
        self.Q[component.cdc_table] = []

    def publish(self, events: t.List[Event]):
        """
        Publish events to local queue.

        :param events: list of events
        """
        with self.lock:
            for event in events:
                self.Q[event.queue].append(event)

        for queue in self.Q:
            consume_events(events=self.Q[queue], table=queue, db=self.db, compute=self.compute)
            self.Q[queue] = []