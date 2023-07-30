from superduperdb.base.encoder import Encoder
from superduperdb.misc.str_shape import str_shape


def vector(shape):
    return Encoder(
        identifier=f'vector[{str_shape(shape)}]',
        shape=shape,
        encoder=None,
        decoder=None,
    )
