from superduperdb.container.encoder import Encoder
from superduperdb.ext.utils import str_shape


def vector(shape):
    """
    Create an encoder for a vector (list of ints/ floats) of a given shape

    :param shape: The shape of the vector
    """
    return Encoder(
        identifier=f'vector[{str_shape(shape)}]',
        shape=shape,
        encoder=None,
        decoder=None,
    )
