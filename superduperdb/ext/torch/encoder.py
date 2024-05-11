import typing as t

import numpy
import torch

from superduperdb.components.datatype import DataType, DataTypeFactory
from superduperdb.ext.utils import str_shape


class EncodeTensor:
    """Encode a tensor to bytes.

    :param dtype: The dtype of the tensor, eg. torch.float32
    """

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, x, info: t.Optional[t.Dict] = None):
        """Encode a tensor to bytes.

        :param x: The tensor to encode.
        :param info: Additional information.
        """
        if x.dtype != self.dtype:
            raise TypeError(f"dtype was {x.dtype}, expected {self.dtype}")
        return memoryview(x.numpy()).tobytes()


class DecodeTensor:
    """Decode a tensor from bytes.

    :param dtype: The dtype of the tensor, eg. torch.float32
    :param shape: The shape of the tensor, eg. (3, 4)
    """

    def __init__(self, dtype, shape):
        self.dtype = torch.randn(1).type(dtype).numpy().dtype
        self.shape = shape

    def __call__(self, bytes, info: t.Optional[t.Dict] = None):
        """Decode a tensor from bytes.

        :param bytes: The bytes to decode.
        :param info: Additional information.
        """
        array = numpy.frombuffer(bytes, dtype=self.dtype).reshape(self.shape)
        return torch.from_numpy(array)


def tensor(dtype, shape: t.Sequence, bytes_encoding: t.Optional[str] = None):
    """
    Create an encoder for a tensor of a given dtype and shape.

    :param dtype: The dtype of the tensor.
    :param shape: The shape of the tensor.
    :param bytes_encoding: The bytes encoding to use.
    """
    return DataType(
        identifier=f"{str(dtype)}[{str_shape(shape)}]",
        encoder=EncodeTensor(dtype),
        decoder=DecodeTensor(dtype, shape),
        shape=shape,
        bytes_encoding=bytes_encoding,
    )


class TorchDataTypeFactory(DataTypeFactory):
    """Factory for torch datatypes.

    It's used for registering the auto schema.
    """

    @staticmethod
    def check(data: t.Any) -> bool:
        """Check if the data is a torch tensor.

        It's used for registering the auto schema.

        :param data: Data to check
        """
        return isinstance(data, torch.Tensor)

    @staticmethod
    def create(data: t.Any) -> DataType:
        """Create a torch tensor datatype.

        It's used for registering the auto schema.

        :param data: Data to create the datatype from
        """
        return tensor(data.dtype, data.shape)
