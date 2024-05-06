import dataclasses as dc
import typing as t

from superduperdb.cdc.cdc import DBEvent, Packet

TokenType = t.Dict[str, str]


@dc.dataclass
class IbisDBPacket(Packet):
    """
    A base packet to represent message in task queue.
    """

    ids: t.List[str]
    query: t.Optional[t.Any] = None
    event_type: DBEvent = DBEvent.insert
