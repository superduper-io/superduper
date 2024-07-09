import typing as t

import numpy

from superduper.components.datatype import DataType, DataTypeFactory
from superduper.ext.utils import str_shape
from superduper.misc.annotations import component


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


@component()
def array(
    dtype: str,
    shape: t.Sequence,
    bytes_encoding: t.Optional[str] = None,
    encodable: str = 'encodable',
):
    """
    Create an encoder of numpy arrays.

    :param dtype: The dtype of the array.
    :param shape: The shape of the array.
    :param bytes_encoding: The bytes encoding to use.
    :param encodable: The encodable to use.
    """
    return DataType(
        identifier=f'numpy-{dtype}[{str_shape(shape)}]',
        encoder=EncodeArray(dtype),
        decoder=DecodeArray(dtype, shape),
        shape=shape,
        bytes_encoding=bytes_encoding,
        encodable=encodable,
    )


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
    def create(data: t.Any) -> DataType:
        """Create a numpy array datatype.

        It's used for registering the auto schema.
        :param data: The numpy array.
        """
        return array(dtype=str(data.dtype), shape=data.shape)
