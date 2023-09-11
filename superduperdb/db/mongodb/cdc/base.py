import dataclasses as dc
import json
import typing as t
from abc import ABC, abstractmethod
from enum import Enum

from bson.objectid import ObjectId as BsonObjectId
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


class DBEvent(Enum):
    """`DBEvent` simple enum to hold mongo basic events."""

    delete: str = 'delete'
    insert: str = 'insert'
    update: str = 'update'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class ObjectId(BsonObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not isinstance(v, BsonObjectId):
            raise TypeError('Id is required.')
        return str(v)


@dc.dataclass
class Packet:
    """
    A base packet to represent message in task queue.
    """

    ids: t.List[t.Union[ObjectId, str]]
    query: t.Optional[Serializable]
    event_type: str = DBEvent.insert.value
