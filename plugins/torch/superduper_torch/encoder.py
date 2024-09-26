import typing as t

import numpy
import torch
from superduper.components.datatype import DataType, DataTypeFactory
from superduper.ext.utils import str_shape
from superduper.misc.annotations import component

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


class EncodeTensor:
    """Encode a tensor to bytes # noqa.

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
    """Decode a tensor from bytes # noqa.

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


@component()
def tensor(
    dtype,
    shape: t.Sequence,
    bytes_encoding: t.Optional[str] = None,
    encodable: str = 'encodable',
    db: t.Optional['Datalayer'] = None,
):
    """Create an encoder for a tensor of a given dtype and shape.

    :param dtype: The dtype of the tensor.
    :param shape: The shape of the tensor.
    :param bytes_encoding: The bytes encoding to use.
    :param encodable: The encodable name
        ["artifact", "encodable", "lazy_artifact", "file"].
    :param db: The datalayer instance.
    """
    dtype = getattr(torch, dtype)
    return DataType(
        identifier=f"{str(dtype).replace('.', '-')}[{str_shape(shape)}]",
        encoder=EncodeTensor(dtype),
        decoder=DecodeTensor(dtype, shape),
        shape=shape,
        bytes_encoding=bytes_encoding,
        encodable=encodable,
    )


class TorchDataTypeFactory(DataTypeFactory):
    """Factory for torch datatypes # noqa.

    This is used for registering the auto schema.
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
        dtype = str(data.dtype).split(".")[1]
        return tensor(dtype=dtype, shape=list(data.shape))
