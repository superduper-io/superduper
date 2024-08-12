import uuid
import dataclasses as dc
import typing as t



@dc.dataclass
class _Component(dict):
    type_id: str
    identifier: str


@dc.dataclass
class Event:
    """Event dataclass to store event data.

    :param type_id: type id of the component.
    :param identifier: Identifier of the component.
    :param id: Id/s of select table.
    :param event_type: Type of the event.
    """

    dest: _Component
    id: t.Any

    src: t.Optional[_Component] = None
    from_type: str = 'DB'
    event_type: str = 'insert'
    uuid: str = str(uuid.uuid4())

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
        db_events = []
        component_events = []
        component_startup_flag = False
        for event in events:
            if event.from_type == 'COMPONENT':
                if component_startup_flag:
                    raise ValueError(
                        'Found {self.type} component initialization job more than once'
                    )
                component_events = [event]
                component_startup_flag = True
            else:
                db_events.append(event)
        return component_events, db_events
