import typing as t

from superduper import logging

DependencyType = t.Union[t.Dict[str, str], t.Sequence[t.Dict[str, str]]]


class LocalSequentialQueue:
    """
    LocalSequentialQueue for handling publisher and consumer process.

    Local queue which holds listeners, vector indices as queue which
    consists of events to be consumed by the corresponding components.
    """

    def __init__(self):
        self.queue = {}
        self.components = {}
        self._db = None
        self._component_map = {}

    def declare_component(self, component):
        """Declare component and add it to queue."""
        identifier = f'{component.type_id}.{component.identifier}'
        self.queue[identifier] = []
        self.components[identifier] = component

    @property
    def db(self):
        """Instance of Datalayer."""
        return self._db

    @db.setter
    def db(self, db):
        self._db = db

    def publish(self, events: t.List[t.Dict], to: DependencyType):
        """
        Publish events to local queue.

        :param events: list of events
        :param to: Component name for events to be published.
        """

        def _publish(events, to):
            identifier = to['identifier']
            type_id = to['type_id']
            self._component_map.update(to)
            identifier = f'{type_id}.{identifier}'
            component = self.components[identifier]

            ready_ids = component.ready_ids([e['identifier'] for e in events])
            ready_events = []
            for event in events:
                id = event['identifier']
                if id in ready_ids:
                    ready_events.append(event)

            self.queue[identifier].extend(ready_events)

        if isinstance(to, (tuple, list)):
            for dep in to:
                _publish(events, dep)
        else:
            _publish(events, to)
        return self.consume()

    def consume(self):
        """Consume the current queue and run jobs."""
        from superduper.base.datalayer import Event

        queue_jobs = {}
        for component_id in self.queue:
            events = self.queue[component_id]
            if not events:
                continue
            self.queue[component_id] = []

            component = self.components[component_id]
            jobs = []

            for event_type, type_events in Event.chunk_by_event(events).items():
                ids = [event['identifier'] for event in type_events]
                overwrite = (
                    True if event_type in [Event.insert, Event.upsert] else False
                )
                logging.info(f'Running jobs for {component_id} with ids: {ids}')
                job = component.run_jobs(
                    db=self.db, ids=ids, overwrite=overwrite, event_type=event_type
                )
                jobs.append(job)

            if component_id in queue_jobs:
                queue_jobs[component_id].extend(jobs)
            else:
                queue_jobs[component_id] = jobs

        return queue_jobs
