import dataclasses as dc
import typing as t
import uuid
from collections import defaultdict
from enum import Enum


@dc.dataclass
class ComponentPlaceholder(dict):
    """Component placeholder to store component data.

    :param type_id: Type id of the component.
    :param identifier: Identifier of the component.
    """

    type_id: str
    identifier: str


class EventType(str, Enum):
    """Event to represent database events.

    # noqa
    """

    insert = 'insert'
    delete = 'delete'
    update = 'update'
    apply = 'apply'


@dc.dataclass
class Event:
    """Event dataclass to store event data.

    :param event_type: Type of the event.
    :param source: Identifier of the destination component.
    :param ids: List of ids for the event.
    :param context: Optional context identifier.
    :param msg: Msg to broadcast.
    """

    event_type: EventType | str
    source: str | None = None
    ids: t.Sequence[str] | None = None
    # TODO uuid needed?
    context: str | None = None
    msg: t.Optional[str] = None

    def dict(self):
        """Convert to dict."""
        return {
            **dc.asdict(self),
            '_path': f'superduper.base.event.{self.__class__.__name__}',
        }

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
            source=self.source,
            ids=list(self.ids) + list(other.ids),
        )

    @staticmethod
    def get_job_ids(events: t.List['Event']):
        """Get job ids from events."""
        ids = []
        for e in events:
            ids.append(e.uuid)
        return ids