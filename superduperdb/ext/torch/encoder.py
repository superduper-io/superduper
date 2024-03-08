import typing as t

import numpy
import torch

from superduperdb.components.datatype import DataType
from superduperdb.ext.utils import str_shape


class EncodeTensor:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, x, info: t.Optional[t.Dict] = None):
        if x.dtype != self.dtype:
            raise TypeError(f'dtype was {x.dtype}, expected {self.dtype}')
        return memoryview(x.numpy()).tobytes()


class DecodeTensor:
    def __init__(self, dtype):
        self.dtype = torch.randn(1).type(dtype).numpy().dtype

    def __call__(self, bytes, info: t.Optional[t.Dict] = None):
        array = numpy.frombuffer(bytes, dtype=self.dtype)
        return torch.from_numpy(array)


def tensor(dtype, shape: t.Sequence):
    """
    Create an encoder for a tensor of a given dtype and shape.

    :param dtype: The dtype of the tensor.
    :param shape: The shape of the tensor.
    """
    return DataType(
        identifier=f'{str(dtype)}[{str_shape(shape)}]',
        encoder=EncodeTensor(dtype),
        decoder=DecodeTensor(dtype),
        shape=shape,
    )
