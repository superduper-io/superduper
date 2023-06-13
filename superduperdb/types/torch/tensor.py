import numpy
import torch

from superduperdb.core.type import Type


class EncodeTensor:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, x):
        assert self.dtype == x.dtype
        return memoryview(x.numpy()).tobytes()


class DecodeTensor:
    def __init__(self, dtype):
        self.dtype = torch.randn(1).type(dtype).numpy().dtype

    def __call__(self, bytes):
        array = numpy.frombuffer(bytes, dtype=self.dtype)
        return torch.from_numpy(array)


def tensor(dtype):
    return Type(
        identifier=str(dtype), encoder=EncodeTensor(dtype), decoder=DecodeTensor(dtype)
    )
