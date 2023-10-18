import dataclasses as dc
import typing as t
from enum import Enum

from superduperdb.container.serializable import Serializable


class DBEvent(str, Enum):
    """`DBEvent` simple enum to hold mongo basic events."""

    delete = 'delete'
    insert = 'insert'
    update = 'update'


@dc.dataclass
class Packet:
    ids: t.Any
    query: t.Optional[Serializable]
    event_type: DBEvent = DBEvent.insert

    @property
    def is_delete(self) -> bool:
        return self.event_type == DBEvent.delete

    @staticmethod
    def collate(packets: t.Sequence['Packet']) -> 'Packet':
        """
        Collate a batch of packets into one
        """
        assert packets
        ids = [packet.ids[0] for packet in packets]
        query = packets[0].query

        # TODO: cluster Packet for each event.
        event_type = packets[0].event_type
        return type(packets[0])(ids=ids, query=query, event_type=event_type)
