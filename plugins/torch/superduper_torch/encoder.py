import dataclasses as dc
import typing as t

import numpy
import torch
from superduper.base.datatype import BaseDataType

if t.TYPE_CHECKING:
    pass


@dc.dataclass(kw_only=True)
class Tensor(BaseDataType):
    """Encode/ decode a numpy array as bytes.

    :param dtype: numpy native datatype.
    :param shape: Shape of array.
    """

    dtype: str = 'float64'
    shape: t.Tuple[int]

    def encode_data(self, item, context):
        """Encode data.

        :param item: item to encode.
        """
        return memoryview(item.numpy()).tobytes()

    def decode_data(self, item, builds, db):
        """Decode data.

        :param item: item to decode.
        """
        array = numpy.frombuffer(item, dtype=self.dtype).reshape(self.shape)
        return torch.from_numpy(array)
