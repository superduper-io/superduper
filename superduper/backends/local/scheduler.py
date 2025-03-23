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
    """Queue wrapper.

    :param identifier: identifier of the queue.
    :param scheduler: LocalScheduler instance.
    """

    def __init__(self, identifier: str, scheduler: 'LocalScheduler'):
        self.scheduler = scheduler
        self.identifier = identifier

    def initialize(self):
        self.scheduler.Q[self.identifier] = []

    def drop(self):
        del self.scheduler.Q[self.identifier]


class LocalScheduler(Bookkeeping, BaseScheduler):
    """
    Class for handling publisher and consumer processes.

    Contains a local queue which holds listeners, vector indices in a queue which
    consists of events to be consumed by the corresponding components.

    :param uri: uri to connect.
    """

    def __init__(self):
        Bookkeeping.__init__(self)
        BaseScheduler.__init__(self)

        self.lock = threading.Lock()
        self.Q: t.Dict = {'_apply': []}

    @property
    def db(self):
        return self._db

    @db.setter
    def db(self, value):
        self._db = value

    def drop(self):
        """Drop the queue."""
        self.Q = {}

    def build_tool(self, component):
        return QueueWrapper(component.cdc_table, self)

    def initialize(self):
        """Initialize the scheduler."""
        # TODO
        # self.Q['_apply'] = []
        # for component_data in self.db.show():
        #     component = component_data['component']
        #     identifier = component_data['identifier']
        #     r = self.db.show(component=component, identifier=identifier, version=-1)
        #     if r.get('cdc_table'):
        #         self.put_component(component)
        #         with self.lock:
        #             self.Q[component, identifier] = []

    def publish(self, events: t.List[Event]):
        """
        Publish events to local queue.

        :param events: list of events
        """
        with self.lock:
            for event in events:
                self.Q[event.queue].append(event)

        queues = list(self.Q.keys())
        for queue in queues:
            events = self.Q[queue]
            self.Q[queue] = []
            consume_events(
                events=events,
                table=queue,
                db=self.db,
            )
