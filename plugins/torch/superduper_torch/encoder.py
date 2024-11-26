import typing as t

import numpy
import torch
from superduper.components.datatype import BaseDataType, DataTypeFactory
from superduper.ext.numpy.encoder import DecodeArray, EncodeArray
from superduper.ext.utils import str_shape

if t.TYPE_CHECKING:
    pass


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
        self.dtype = torch.from_numpy(numpy.random.randn(1).astype(dtype)).dtype
        self.shape = shape

    def __call__(self, bytes, info: t.Optional[t.Dict] = None):
        """Decode a tensor from bytes.

        :param bytes: The bytes to decode.
        :param info: Additional information.
        """
        array = numpy.frombuffer(bytes, dtype=self.dtype).reshape(self.shape)
        return torch.from_numpy(array)


class Tensor(BaseDataType):
    """Encode/ decode a numpy array as bytes.

    :param dtype: numpy native datatype.
    :param shape: Shape of array.
    """

    dtype: str = 'float64'
    shape: int | t.Tuple[int]
    identifier: str = ''

    def __post_init__(self, db):
        self.encodable = 'encodable'
        if not self.identifier:
            dtype = str(self.dtype)
            self.identifier = f'torch-{dtype}[{str_shape(self.shape)}]'
        return super().__post_init__(db)

    def encode_data(self, item):
        """Encode data.

        :param item: item to encode.
        """
        encoder = EncodeArray(self.dtype)
        return encoder(item.numpy())

    def decode_data(self, item):
        """Decode data.

        :param item: item to decode.
        """
        shape = self.shape
        if isinstance(shape, int):
            shape = (self.shape,)
        decoder = DecodeArray(self.dtype, shape=shape)
        return torch.from_numpy(decoder(item))


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
    def create(data: t.Any) -> BaseDataType:
        """Create a torch tensor datatype.

        It's used for registering the auto schema.

        :param data: Data to create the datatype from
        """
        dtype = str(data.dtype).split(".")[1]
        return Tensor(dtype=dtype, shape=list(data.shape))
