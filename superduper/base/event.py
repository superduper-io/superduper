from collections import defaultdict
import dataclasses as dc
from enum import Enum
import typing as t
import uuid


@dc.dataclass
class ComponentPlaceholder(dict):
    """Component placeholder to store component data.

    :param type_id: Type id of the component.
    :param identifier: Identifier of the component.
    """
    type_id: str
    identifier: str



class EventType(str, Enum):
    """Event to represent database events. # noqa """
    insert = 'insert'
    delete = 'delete'
    update = 'update'
    upsert = 'upsert'     # TODO why do we need this?
    apply = 'apply'


@dc.dataclass
class Event:
    """Event dataclass to store event data.

    :param event_type: Type of the event.
    :param dest: Identifier of the destination component.
    :param ids: List of ids for the event.
    :param uuid: Unique identifier for the event.
                 This id will be used as job id in
                 startup events.
    """
    event_type: EventType
    dest: t.Union[ComponentPlaceholder, t.Dict]
    ids: t.Sequence[str] | None = None
    uuid: str = dc.field(default_factory=lambda: str(uuid.uuid4()).replace('-', ''))

    def dict(self):
        """Convert to dict."""
        return {**dc.asdict(self), '_path': f'superduper.base.event.{self.__class__.__name__}'}

    def __add__(self, other: 'Event'):
        """Add two events."""
        if self.event_type == 'apply':
            assert self.ids is None
            assert other.ids is None
            return self
        assert self.event_type != 'apply'
        assert self.ids is not None
        assert other.ids is not None
        r = self.dict()
        s = other.dict()
        for k in r.keys():
            if k not in {'ids', 'uuid'}:
                assert r[k] == s[k]
        return Event(
            event_type=self.event_type,
            dest=self.dest,
            ids=self.ids + other.ids,
        )

    @staticmethod
    def get_job_ids(events: t.List['Event']):
        """Get job ids from events."""
        ids = []
        for e in events:
            ids.append(e.uuid)
        return ids

    @staticmethod
    def chunk_by_type(events: t.Sequence['Event']): 
        """Chunk events by from type."""
        out = defaultdict(list)
        for event in events:
            out[event.event_type].append(event)
        chunked = {}
        for k in out:
            chunked[k] = out[k][0]
            for i in range(1, len(out[k])):
                chunked[k] += out[k][i]
        return chunked
