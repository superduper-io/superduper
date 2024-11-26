import typing as t

import numpy

from superduper.components.datatype import (
    BaseDataType,
    DataTypeFactory,
)
from superduper.ext.utils import str_shape


class EncodeArray:
    """Encode a numpy array to bytes.

    :param dtype: The dtype of the array.
    """

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, x, info: t.Optional[t.Dict] = None):
        """Encode the numpy array to bytes.

        :param x: The numpy array.
        :param info: The info of the encoding.
        """
        if x.dtype != self.dtype:
            raise TypeError(f'dtype was {x.dtype}, expected {self.dtype}')
        return memoryview(x).tobytes()


class DecodeArray:
    """Decode a numpy array from bytes.

    :param dtype: The dtype of the array.
    :param shape: The shape of the array.
    """

    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape

    def __call__(self, bytes, info: t.Optional[t.Dict] = None):
        """Decode the numpy array from bytes.

        :param bytes: The bytes to decode.
        :param info: The info of the encoding.
        """
        return numpy.frombuffer(bytes, dtype=self.dtype).reshape(self.shape)


class Array(BaseDataType):
    """Encode/ decode a numpy array as bytes.

    :param dtype: numpy native datatype.
    :param shape: Shape of array.
    """

    dtype: str = 'float64'
    shape: int | t.Tuple[int]
    identifier: str = ''

    def __post_init__(self, db):
        self.encodable = 'encodable'
        if not self.identifier:
            dtype = str(self.dtype)
            self.identifier = f'numpy-{dtype}[{str_shape(self.shape)}]'
        return super().__post_init__(db)

    def encode_data(self, item):
        encoder = EncodeArray(self.dtype)
        return encoder(item)

    def decode_data(self, item):
        shape = self.shape
        if isinstance(shape, int):
            shape = (self.shape,)
        decoder = DecodeArray(self.dtype, shape=shape)
        return decoder(item)


class NumpyDataTypeFactory(DataTypeFactory):
    """A factory for numpy arrays # noqa."""

    @staticmethod
    def check(data: t.Any) -> bool:
        """Check if the data is a numpy array.

        It's used for registering the auto schema.
        :param data: The data to check.
        """
        return isinstance(data, numpy.ndarray)

    @staticmethod
    def create(data: t.Any) -> Array:
        """Create a numpy array datatype.

        It's used for registering the auto schema.
        :param data: The numpy array.
        """
        return Array(dtype=str(data.dtype), shape=list(data.shape))
