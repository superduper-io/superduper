from superduperdb.misc import dataclasses as dc
import dataclasses
import io
import pickle
import typing as t

from superduperdb.core.artifact import Artifact
from superduperdb.core.component import Component

Decode = t.Callable[[bytes], t.Any]
Encode = t.Callable[[t.Any], bytes]


def _pickle_decoder(b: bytes) -> t.Any:
    return pickle.load(io.BytesIO(b))


def _pickle_encoder(x: t.Any) -> bytes:
    f = io.BytesIO()
    pickle.dump(x, f)
    return f.getvalue()


@dataclasses.dataclass
class Encoder(Component):
    """
    Storeable ``Component`` allowing byte encoding of primary data,
    i.e. data inserted using ``datalayer.base.BaseDatabase.insert``

    :param identifier: unique identifier
    :param encoder: callable converting an ``Encodable`` of this ``Encoder`` to
                    be converted to ``bytes``
    :param decoder: callable converting a ``bytes`` string to a ``Encodable`` of
                    this ``Encoder``
    :param shape: shape of the data, if any
    """

    variety: t.ClassVar[str] = 'type'  # This cannot yet be changed
    artifacts: t.ClassVar[t.List[str]] = ['decoder', 'encoder']

    identifier: str
    decoder: t.Union[t.Callable, Artifact] = Artifact(_artifact=_pickle_decoder)
    encoder: t.Union[t.Callable, Artifact] = Artifact(_artifact=_pickle_encoder)
    shape: t.Optional[t.Tuple] = None
    version: t.Optional[int] = None

    def __post_init__(self):
        if isinstance(self.decoder, t.Callable):
            self.decoder = Artifact(_artifact=self.decoder)
        if isinstance(self.encoder, t.Callable):
            self.encoder = Artifact(_artifact=self.encoder)

    def __call__(self, x):
        return Encodable(x, self)

    def decode(self, b: bytes) -> t.Any:
        return self(self.decoder.a(b))

    def encode(self, x: t.Any) -> t.Dict[str, t.Any]:
        if self.encoder is not None:
            return {'_content': {'bytes': self.encoder.a(x), 'type': self.identifier}}
        else:
            return x


@dc.dataclass
class Encodable:
    """
    Data variable wrapping encode-able item. Encoding is controlled by the referred
    to ``Encoder`` instance.

    :param x: Wrapped content
    :param encoder: Instance of ``Encoder`` controlling encoding
    """

    x: t.Any
    encoder: t.Callable

    def encode(self) -> t.Dict[str, t.Any]:
        return self.encoder.encode(self.x)


default_encoder = Encoder(identifier='_default')
