import numpy
import torch


class Array:
    def __init__(self, dtype, types=()):
        self.dtype = torch.randn(1).type(dtype).numpy().dtype
        self.types = types

    def encode(self, x):
        x = x.numpy()
        assert self.dtype == x.dtype
        return memoryview(x).tobytes()

    def decode(self, bytes_):
        array = numpy.frombuffer(bytes_, dtype=self.dtype)
        return torch.from_numpy(array)
