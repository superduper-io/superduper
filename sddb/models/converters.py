import io
import numpy
import PIL.Image
import torch

from sddb.utils import import_object

converters = {}


def encode(handle, bytes_):
    if handle not in converters:
        converters[handle] = import_object(handle)
    return converters[handle].encode(bytes_)


def decode(handle, bytes_):
    if handle not in converters:
        converters[handle] = import_object(handle)
    return converters[handle].decode(bytes_)


class PILImage:
    @staticmethod
    def encode(x):
        buffer = io.BytesIO()
        x.save(buffer, format='png')
        return buffer.getvalue()

    @staticmethod
    def decode(bytes_):
        return PIL.Image.open(io.BytesIO(bytes_))


class FloatTensor:
    @staticmethod
    def encode(x):
        x = x.numpy()
        assert x.dtype == numpy.float32
        return memoryview(x).tobytes()

    @staticmethod
    def decode(bytes_):
        array = numpy.frombuffer(bytes_, dtype=numpy.float32)
        return torch.from_numpy(array).type(torch.float)
