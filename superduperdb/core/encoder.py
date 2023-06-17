import io
import pickle
import typing as t

from .data_var import DataVar
from superduperdb.core.base import Component


class Encoder(Component):
    """
    Storeable ``Component`` allowing byte encoding of primary data,
    i.e. data inserted using ``datalayer.base.BaseDatabase._insert``

    :param identifier: unique identifier
    :param encoder: callable converting an ``DataVar`` of this ``Encoder`` to
                    be converted to ``bytes``
    :param decoder: callable converting a ``bytes`` string to a ``DataVar`` of
                    this ``Encoder``
    """

    type_id = 'type'  # This cannot yet be changed

    def __init__(
        self,
        identifier,
        encoder=None,
        decoder=None,
        shape: t.Optional[t.Tuple] = None,
    ):
        super().__init__(identifier)
        self.encoder = encoder or _pickle_dump
        self.decoder = decoder or _pickle_load
        self.shape = shape

    def __repr__(self):
        if self.shape is not None:
            return f'Encoder[{self.identifier}/{self.version}:{tuple(self.shape)}]'
        else:
            return f'Encoder[{self.identifier}/{self.version}]'

    def decode(self, b: bytes) -> t.Any:
        return DataVar(self.decoder(b), type=self.identifier, encoder=self.encoder)

    def __call__(self, x):
        return DataVar(
            x,
            type=self.identifier,
            encoder=self.encoder,
            shape=self.shape,
        )


def _pickle_load(x: bytes) -> t.Any:
    return pickle.load(io.BytesIO(b))


def _pickle_dump(x: t.Any) -> bytes:
    f = io.BytesIO()
    pickle.dump(x, f)
    return f.getvalue()
