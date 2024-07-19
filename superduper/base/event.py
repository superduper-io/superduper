import dataclasses as dc
import typing as t


@dc.dataclass
class Event:
    """Event dataclass to store event data.

    :param type_id: type id of the component.
    :param identifier: Identifier of the component.
    :param id: Id of select table.
    :param event_type: Type of the event.
    """

    type_id: str
    identifier: str
    id: t.Any
    event_type: str

    def dict(self):
        """Convert to dict."""
        return dc.asdict(self)
