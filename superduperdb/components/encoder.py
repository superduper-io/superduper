import dataclasses as dc
import io
import pickle
import typing as t

from superduperdb.base.artifact import Artifact
from superduperdb.components.component import Component

Decode = t.Callable[[bytes], t.Any]
Encode = t.Callable[[t.Any], bytes]


def _pickle_decoder(b: bytes) -> t.Any:
    return pickle.load(io.BytesIO(b))


def _pickle_encoder(x: t.Any) -> bytes:
    f = io.BytesIO()
    pickle.dump(x, f)
    return f.getvalue()


@dc.dataclass
class Encoder(Component):
    """
    Storeable ``Component`` allowing byte encoding of primary data,
    i.e. data inserted using ``db.base.db.Datalayer.insert``

    :param identifier: Unique identifier
    :param decoder: callable converting a ``bytes`` string to a ``Encodable``
                    of this ``Encoder``
    :param encoder: Callable converting an ``Encodable`` of this ``Encoder``
                    to ``bytes``
    :param shape: Shape of the data
    :param version: Version of the encoder (don't use this)
    :param load_hybrid: Whether to load the data from the URI or return the URI in
                        `CFG.hybrid` mode
    """

    artifact_artibutes: t.ClassVar[t.Sequence[str]] = ['decoder', 'encoder']
    identifier: str
    decoder: t.Union[t.Callable, Artifact] = dc.field(
        default_factory=lambda: Artifact(artifact=_pickle_decoder)
    )
    encoder: t.Union[t.Callable, Artifact] = dc.field(
        default_factory=lambda: Artifact(artifact=_pickle_encoder)
    )
    shape: t.Optional[t.Sequence] = None
    load_hybrid: bool = True

    # Don't set this manually
    version: t.Optional[int] = None
    type_id: t.ClassVar[str] = 'encoder'
    # TODO what's this for?
    encoders: t.ClassVar[t.List] = []

    def __post_init__(self):
        self.encoders.append(self.identifier)
        if isinstance(self.decoder, t.Callable):
            self.decoder = Artifact(artifact=self.decoder)
        if isinstance(self.encoder, t.Callable):
            self.encoder = Artifact(artifact=self.encoder)

    def __call__(
        self, x: t.Optional[t.Any] = None, uri: t.Optional[str] = None
    ) -> 'Encodable':
        return Encodable(self, x=x, uri=uri)

    def decode(self, b: bytes) -> t.Any:
        assert isinstance(self.decoder, Artifact)
        return self(self.decoder.artifact(b))

    def dump(self, other):
        return self.encoder.artifact(other)

    def encode(
        self,
        x: t.Optional[t.Any] = None,
        uri: t.Optional[str] = None,
        wrap: bool = True,
    ) -> t.Union[t.Optional[str], t.Dict[str, t.Any]]:
        # TODO clarify what is going on here
        def _wrap_content(x):
            return {
                '_content': {
                    'bytes': self.encoder.artifact(x),
                    'encoder': self.identifier,
                }
            }

        if self.encoder is not None:
            if x is not None:
                if wrap:
                    return _wrap_content(x)
                return self.encoder.artifact(x)  # type: ignore[union-attr]
            else:
                if wrap:
                    return {
                        '_content': {
                            'uri': uri,
                            'encoder': self.identifier,
                        }
                    }
                return uri
        else:
            assert x is not None
            return x


@dc.dataclass
class Encodable:
    """
    Data variable wrapping encode-able item. Encoding is controlled by the referred
    to ``Encoder`` instance.

    :param encoder: Instance of ``Encoder`` controlling encoding
    :param x: Wrapped content
    :param uri: URI of the content, if any
    """

    encoder: Encoder
    x: t.Optional[t.Any] = None
    uri: t.Optional[str] = None

    def encode(self) -> t.Union[t.Optional[str], t.Dict[str, t.Any]]:
        assert hasattr(self.encoder, 'encode')
        return self.encoder.encode(x=self.x, uri=self.uri)


default_encoder = Encoder(identifier='_default')
