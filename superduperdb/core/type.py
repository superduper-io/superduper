import io
import pickle
import typing as t

from .data_var import DataVar
from superduperdb.core.base import Component


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
