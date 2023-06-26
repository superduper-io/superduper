import numpy
import typing as t

from superduperdb.core.encoder import Encoder
from superduperdb.types.utils import str_shape


class EncodeArray:
    def __init__(self, dtype) -> None:
        self.dtype = dtype

    def __call__(self, x) -> bytes:
        assert x.dtype == self.dtype
        return memoryview(x).tobytes()


class DecodeArray:
    def __init__(self, dtype) -> None:
        self.dtype = dtype

    def __call__(self, bytes) -> numpy.ndarray:
        return numpy.frombuffer(bytes, dtype=self.dtype)


def array(dtype: str, shape: t.Tuple) -> Encoder:
    return Encoder(
        identifier=f'numpy.{dtype}[{str_shape(shape)}]',
        encoder=EncodeArray(dtype),
        decoder=DecodeArray(dtype),
        shape=shape,
    )
