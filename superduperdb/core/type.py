import io
import pickle
import typing as t

from superduperdb.core.base import Component


class DataVar:
    """
    Data variable wrapping encode-able item. Encoding is controlled by the referred
    to ``Type`` instance.

    :param x: Wrapped content
    :param type: Identifier of type component used to encode
    :param encoder: Encoder used to dump to `bytes`
    """

    def __init__(
        self,
        x: t.Any,
        type: str,
        encoder: t.Optional[t.Callable] = None,
        shape: t.Optional[t.Tuple] = None,
    ):
        if shape is not None:
            assert hasattr(x, 'shape')
            assert tuple(x.shape) == shape
        self.x = x
        self._encoder = encoder
        self.type = type
        self.shape = shape

    def __repr__(self):
        if self.shape is not None:
            return f'DataVar[{self.type}: {tuple(self.shape)}]({self.x.__repr__()})'
        else:
            return f'DataVar[{self.type}]({self.x.__repr__()})'

    def encode(self):
        if self._encoder is None:
            f = io.BytesIO()
            pickle.dump(self.x, f)
            return f.getvalue()
        return {'_content': {'bytes': self._encoder(self.x), 'type': self.type}}


class Type(Component):
    """
    Storeable ``Component`` allowing byte encoding of primary data,
    i.e. data inserted using ``datalayer.base.BaseDatabase._insert``

    :param identifier: unique identifier
    :param encoder: callable converting an ``DataVar`` of this ``Type`` to
                    be converted to ``bytes``
    :param decoder: callable converting a ``bytes`` string to a ``DataVar`` of
                    this ``Type``
    """

    type_id = 'type'

    def __init__(
        self,
        identifier,
        encoder=None,
        decoder=None,
        shape: t.Optional[t.Tuple] = None,
    ):
        super().__init__(identifier)
        self.encoder = encoder
        self.decoder = decoder
        self.shape = shape

    def __repr__(self):
        if self.shape is not None:
            return f'Type[{self.identifier}/{self.version}:{tuple(self.shape)}]'
        else:
            return f'Type[{self.identifier}/{self.version}]'

    def decode(self, bytes):
        if self.decoder is None:
            return DataVar(
                pickle.load(io.BytesIO(bytes)),
                type=self.identifier,
                encoder=self.encoder,
            )
        return DataVar(
            self.decoder(bytes),
            type=self.identifier,
            encoder=self.encoder,
            shape=self.shape,
        )

    def __call__(self, x):
        return DataVar(
            x,
            type=self.identifier,
            encoder=self.encoder,
            shape=self.shape,
        )
