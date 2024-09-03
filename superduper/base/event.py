import dataclasses as dc
import typing as t
import uuid


@dc.dataclass
class _Component(dict):
    type_id: str
    identifier: str


@dc.dataclass
class Event:
    """Event dataclass to store event data.

    :param dest: Identifier of the destination component.
    :param id: Id/s of select table.
    :param method: Method to be called on the destination
    :param src: Identifier of the source component.
    :param from_type: 'COMPONENT' or 'DB' type implying
                      the event was created from a databas
                      e or component event (initlization).

    :param event_type: Type of the event.
    :param uuid: Unique identifier for the event.
                 This id will be used as job id in
                 startup events.
    :param dependencies: List of dependencies on the event.
    """

    dest: t.Union[_Component, t.Dict]
    id: t.Optional[t.Any] = None
    method: t.Optional[str] = None
    # where is this src attribute used?
    src: t.Optional[_Component] = None
    from_type: str = 'DB'
    event_type: str = 'insert'
    uuid: str = dc.field(default_factory=lambda: str(uuid.uuid4()).replace('-', ''))
    dependencies: t.Optional[t.Sequence] = ()

    def __post_init__(self):
        if not self.src:
            self.src = self.dest

    def dict(self):
        """Convert to dict."""
        return dc.asdict(self)

    @staticmethod
    def get_job_ids(events: t.List['Event']):
        """Get job ids from events."""
        ids = []
        for e in events:
            ids.append(e.uuid)
        return ids

    @staticmethod
    def chunk_by_type(events):
        """Chunk events by from type."""
        db_events = []
        component_events = []
        for event in events:
            if event.from_type == 'COMPONENT':
                component_events = [event]
            else:
                db_events.append(event)
        return component_events, db_events
