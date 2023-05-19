import numpy


class Array:
    """
    >>> x = numpy.random.randn(32)
    >>> bs = Array(numpy.float64).encode(x)
    >>> Array(numpy.float64).decode(bs) == x
    True
    """

    def __init__(self, dtype, types=()):
        self.types = types
        self.dtype = dtype

    def encode(self, x):
        assert x.dtype == self.dtype
        return memoryview(x).tobytes()

    def decode(self, bytes_):
        return numpy.frombuffer(bytes_, dtype=self.dtype)
