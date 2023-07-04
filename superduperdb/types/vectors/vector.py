from superduperdb.core.encoder import Encoder
from superduperdb.types.utils import str_shape


def vector(shape):
    return Encoder(
        identifier=f'vector[{str_shape(shape)}]',
        shape=shape,
        encoder=None,
        decoder=None,
    )
