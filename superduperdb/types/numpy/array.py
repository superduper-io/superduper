import numpy

from superduperdb.core.type import Type


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


def array(dtype: str):
    return Type(
        identifier=f'numpy.{dtype}',
        encoder=EncodeArray(dtype),
        decoder=DecodeArray(dtype),
    )
