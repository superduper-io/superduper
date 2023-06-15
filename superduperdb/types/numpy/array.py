import numpy
import typing as t

from superduperdb.core.type import Type
from superduperdb.types.utils import str_shape


class EncodeArray:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, x):
        assert x.dtype == self.dtype
        return memoryview(x).tobytes()


class DecodeArray:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, bytes):
        return numpy.frombuffer(bytes, dtype=self.dtype)


def array(dtype: str, shape: t.Tuple):
    return Type(
        identifier=f'numpy.{dtype}[{str_shape(shape)}]',
        encoder=EncodeArray(dtype),
        decoder=DecodeArray(dtype),
        shape=shape,
    )
