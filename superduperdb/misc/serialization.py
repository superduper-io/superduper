import io
import pickle
from abc import ABC
import dill
import torch
import typing as t
import typing_extensions as te

from superduperdb.models.torch.utils import device_of
from superduperdb.core.serializable import Serializable

Info = t.Optional[t.Dict[str, t.Any]]


class ModuleClassDict(te.TypedDict):
    """A ``dict``with the module, class and JSONization of an object"""

    module: str
    cls: str
    dict: t.Dict[str, t.Any]


class Serializer(ABC):
    @staticmethod
    def encode(object: t.Any, info: Info = None) -> bytes:
        raise NotImplementedError

    @staticmethod
    def decode(b: bytes, info: Info = None) -> t.Any:
        raise NotImplementedError


class PickleSerializer(Serializer):
    @staticmethod
    def encode(object: t.Any, info: Info = None) -> bytes:
        return pickle.dumps(object)

    @staticmethod
    def decode(b: bytes, info: Info = None) -> t.Any:
        return pickle.loads(b)


class DillSerializer(Serializer):
    @staticmethod
    def encode(object: t.Any, info: Info = None) -> bytes:
        return dill.dumps(object)

    @staticmethod
    def decode(b: bytes, info: Info = None) -> t.Any:
        return dill.loads(b)


class TorchSerializer(Serializer):
    @staticmethod
    def encode(object: t.Any, info: Info = None) -> bytes:
        was_gpu = str(device_of(object)) == 'cuda'
        object.to('cpu')
        f = io.BytesIO()
        torch.save(object, f)
        if was_gpu:
            object.to('cuda')
        return f.getvalue()

    @staticmethod
    def decode(b: bytes, info: Info = None) -> t.Any:
        return torch.load(io.BytesIO(b))


class TorchStateSerializer(Serializer):
    @staticmethod
    def encode(object: t.Any, info: Info = None) -> bytes:
        f = io.BytesIO()
        torch.save(object.state_dict(), f)
        return f.getvalue()

    @staticmethod
    def decode(bytes, info):
        instance = Serializable.deserialize(info)
        object = torch.load(io.BytesIO(bytes))
        instance.load_state_dict(object)
        return instance


serializers: t.Dict[str, t.Type] = {
    'pickle': PickleSerializer,
    'dill': DillSerializer,
    'torch': TorchSerializer,
    'torch::state': TorchStateSerializer,
}
