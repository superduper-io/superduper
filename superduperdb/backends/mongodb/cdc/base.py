import dataclasses as dc
import json
import typing as t

from bson import objectid

from superduperdb.base.serializable import Serializable
from superduperdb.cdc.cdc import DBEvent, Packet

TokenType = t.Dict[str, str]


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
class MongoDBPacket(Packet):
    """
    A base packet to represent message in task queue.
    """

    ids: t.List[t.Union[ObjectId, str]]
    query: t.Optional[Serializable] = None
    event_type: DBEvent = DBEvent.insert
