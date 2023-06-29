import typing as t

import numpy
import torch
from torch import Tensor

from superduperdb.core.encoder import Encoder
from superduperdb.types.utils import str_shape


class EncodeTensor:
    def __init__(self, dtype: str) -> None:
        self.dtype = dtype

    def __call__(self, x: Tensor) -> bytes:
        assert self.dtype == x.dtype
        return memoryview(x.numpy()).tobytes()


class DecodeTensor:
    def __init__(self, dtype: str) -> None:
        self.dtype = torch.randn(1).type(dtype).numpy().dtype

    def __call__(self, bytes: bytes) -> Tensor:
        array = numpy.frombuffer(bytes, dtype=self.dtype)
        return torch.from_numpy(array)


def tensor(dtype: str, shape: t.Tuple[int]) -> Encoder:
    return Encoder(
        identifier=f'{str(dtype)}[{str_shape(shape)}]',
        encoder=EncodeTensor(dtype),
        decoder=DecodeTensor(dtype),
        shape=shape,
    )
