from functools import wraps
import dataclasses as dc
import io
import pickle
import typing as t

from superduperdb.core.base import Component

Decode = t.Callable[[bytes], t.Any]
Encode = t.Callable[[t.Any], bytes]


def _pickle_decoder(b: bytes) -> t.Any:
    return pickle.load(io.BytesIO(b))


def _pickle_encoder(x: t.Any) -> bytes:
    f = io.BytesIO()
    pickle.dump(x, f)
    return f.getvalue()


@dc.dataclass
class TypeDesc:
    identifier: str
    decoder: Decode = _pickle_decoder
    encoder: Encode = _pickle_encoder
    shape: t.Optional[t.Tuple] = None

    variety = 'type'


class Type(Component, TypeDesc):
    """
    Storeable ``Component`` allowing byte encoding of primary data,
    i.e. data inserted using ``datalayer.base.BaseDatabase._insert``

    :param identifier: unique identifier
    :param encoder: callable converting an ``DataVar`` of this ``Type`` to
                    be converted to ``bytes``
    :param decoder: callable converting a ``bytes`` string to a ``DataVar`` of
                    this ``Type``
    """

    @wraps(TypeDesc.__init__)
    def __init__(self, identifier, *a, **ka):
        Component.__init__(self, identifier)
        TypeDesc.__init__(self, identifier, *a, **ka)

    def __call__(self, x):
        return DataVar(x, self)

    def decode(self, b: bytes) -> t.Any:
        return self(self.decoder(b))

    def encode(self, x: t.Any) -> t.Dict[str, t.Any]:
        return {'_content': {'bytes': self.encoder(x), 'type': self.identifier}}


@dc.dataclass
class DataVar:
    """
    Data variable wrapping encode-able item. Encoding is controlled by the referred
    to ``Type`` instance.

    :param x: Wrapped content
    :param type: Identifier of type component used to encode
    """

    x: t.Any
    type: Type

    def encode(self) -> t.Dict[str, t.Any]:
        return self.type.encode(self.x)
