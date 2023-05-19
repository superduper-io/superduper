import io
import numpy
import PIL.Image
import PIL.JpegImagePlugin
import torch


class Array32:
    types = [numpy.ndarray]

    @staticmethod
    def encode(x):
        assert x.dtype == numpy.float32
        return memoryview(x).tobytes()

    @staticmethod
    def decode(bytes_):
        return numpy.frombuffer(bytes_, dtype=numpy.float32)


class Int64:
    types = [numpy.int64]

    @staticmethod
    def encode(x):
        assert x.dtype == numpy.int64
        return memoryview(x).tobytes()

    @staticmethod
    def decode(bytes_):
        return numpy.frombuffer(bytes_, dtype=numpy.int64)


class FloatTensor:
    types = [torch.FloatTensor, torch.Tensor]

    @staticmethod
    def encode(x):
        x = x.numpy()
        assert x.dtype == numpy.float32
        return memoryview(x).tobytes()

    @staticmethod
    def decode(bytes_):
        array = numpy.frombuffer(bytes_, dtype=numpy.float32)
        return torch.from_numpy(array).type(torch.float)


class Image:
    types = (PIL.JpegImagePlugin.JpegImageFile,)

    @staticmethod
    def encode(x):
        buffer = io.BytesIO()
        x.save(buffer, format='png')
        return buffer.getvalue()

    @staticmethod
    def decode(bytes_):
        return PIL.Image.open(io.BytesIO(bytes_))
