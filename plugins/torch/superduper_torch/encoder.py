import typing as t

import numpy
import torch
from superduper.components.datatype import BaseDataType, DataTypeFactory
from superduper.misc.utils import str_shape

if t.TYPE_CHECKING:
    pass


class Tensor(BaseDataType):
    """Encode/ decode a numpy array as bytes.

    :param dtype: numpy native datatype.
    :param shape: Shape of array.
    """

    dtype: str = 'float64'
    shape: int | t.Tuple[int]
    identifier: str = ''

    def postinit(self):
        """Post-initialization method."""
        self.encodable = 'encodable'
        if not self.identifier:
            dtype = str(self.dtype)
            self.identifier = f'torch-{dtype}[{str_shape(self.shape)}]'
        return super().postinit()

    def encode_data(self, item):
        """Encode data.

        :param item: item to encode.
        """
        return memoryview(item.numpy()).tobytes()

    def decode_data(self, item):
        """Decode data.

        :param item: item to decode.
        """
        array = numpy.frombuffer(item, dtype=self.dtype).reshape(self.shape)
        return torch.from_numpy(array)


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
