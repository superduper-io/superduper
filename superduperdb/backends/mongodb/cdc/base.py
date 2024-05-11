import dataclasses as dc
import json
import os
import typing as t

from bson import objectid

from superduperdb.base.serializable import Serializable
from superduperdb.cdc.cdc import DBEvent, Packet

TokenType = t.Dict[str, str]


class CachedTokens:
    """A class to cache the CDC tokens in a file.

    This class is used to cache the CDC tokens in a file.
    """

    token_path = os.path.join('.superduperdb', '.cdc.tokens')
    separate = '\n'

    def __init__(self):
        # BROKEN: self._current_tokens is never read from
        self._current_tokens = []
        os.makedirs('.superduperdb', exist_ok=True)

    def append(self, token: TokenType) -> None:
        """Append the token to the file.

        :param token: The token to be appended.
        """
        with open(CachedTokens.token_path, 'a') as fp:
            stoken = json.dumps(token)
            stoken = stoken + self.separate
            fp.write(stoken)

    def load(self) -> t.Sequence[TokenType]:
        """Load the tokens from the file."""
        with open(CachedTokens.token_path) as fp:
            tokens = fp.read().split(self.separate)[:-1]
            self._current_tokens = [TokenType(json.loads(t)) for t in tokens]
        return self._current_tokens


class ObjectId(objectid.ObjectId):
    """A class to represent the ObjectId.

    This class is a subclass of the `bson.objectid.ObjectId` class.
    Use this class to validate the ObjectId.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        """Validate the ObjectId.

        :param v: The value to be validated.
        """
        if not isinstance(v, objectid.ObjectId):
            raise TypeError('Id is required.')
        return str(v)


@dc.dataclass
class MongoDBPacket(Packet):
    """A base packet to represent message in task queue.

    This class is a subclass of the `Packet` class.

    :param ids: The ids of the rows.
    :param query: The query to be executed.
    :param event_type: The event type.
    """

    ids: t.List[t.Union[ObjectId, str]]
    query: t.Optional[Serializable] = None
    event_type: DBEvent = DBEvent.insert
