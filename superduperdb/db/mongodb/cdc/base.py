import dataclasses as dc
import json
import typing as t
from abc import ABC, abstractmethod
from enum import Enum

from bson import objectid
from pymongo.change_stream import CollectionChangeStream

from superduperdb.container.serializable import Serializable

TokenType = t.Dict[str, str]


class BaseDatabaseListener(ABC):
    """
    A Base class which defines basic functions to implement.
    """

    @abstractmethod
    def listen(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def setup_cdc(self) -> CollectionChangeStream:
        pass

    @abstractmethod
    def next_cdc(self, stream: CollectionChangeStream) -> None:
        pass


class CachedTokens:
    token_path = '.cdc.tokens'
    separate = '\n'

    def __init__(self):
        # BROKEN: self._current_tokens is never read from
        self._current_tokens = []

    def append(self, token: TokenType) -> None:
        with open(CachedTokens.token_path, 'a') as fp:
            stoken = json.dumps(token)
            stoken = stoken + self.separate
            fp.write(stoken)

    def load(self) -> t.Sequence[TokenType]:
        with open(CachedTokens.token_path) as fp:
            tokens = fp.read().split(self.separate)[:-1]
            self._current_tokens = [TokenType(json.loads(t)) for t in tokens]
        return self._current_tokens


class DBEvent(str, Enum):
    """`DBEvent` simple enum to hold mongo basic events."""

    delete = 'delete'
    insert = 'insert'
    update = 'update'


class ObjectId(objectid.ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not isinstance(v, objectid.ObjectId):
            raise TypeError('Id is required.')
        return str(v)


@dc.dataclass
class Packet:
    """
    A base packet to represent message in task queue.
    """

    ids: t.List[t.Union[ObjectId, str]]
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

        ids = [packet.ids[0] for packet in packets]
        query = packets[0].query

        # TODO: cluster Packet for each event.
        event_type = packets[0].event_type
        return Packet(ids=ids, query=query, event_type=event_type)
