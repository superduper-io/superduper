import importlib
import io
import pickle
from abc import ABC
import dill
import torch
import typing as t
import typing_extensions as te

Info = t.Optional[t.Dict[str, t.Any]]


class ModuleClassDict(te.TypedDict):
    """A ``dict``with the module, class and JSONization of an object"""

    module: str
    cls: str
    dict: t.Dict[str, t.Any]


def to_dict(item: t.Any) -> ModuleClassDict:
    """Convert an item into a ``ModuleClassDict``

    :param item
        The item to be converted
    """
    return ModuleClassDict(
        module=item.__class__.__module__,
        cls=item.__class__.__name__,
        dict=item.dict(),
    )


def from_dict(r: ModuleClassDict, **kwargs) -> t.Any:
    """Construct an item from a ``ModuleClassDict``

    :param r
        A ModuleClassDict with the module, class and JSONization of an object
    """
    module = importlib.import_module(r['module'])
    cls = getattr(module, r['cls'])
    return cls(**r['dict'], **kwargs)


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
        was_gpu = object.device == 'cuda'
        object.to('cpu')
        f = io.BytesIO()
        object.save(f)
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
    def decode(object: t.Any, info: Info = None) -> t.Any:
        # BROKEN: decode's first argument is bytes, not t.Any!
        instance = from_dict(info)  # type: ignore[arg-type]
        instance.load_state_dict(object)
        return instance


serializers: t.Dict[str, t.Type] = {
    'pickle': PickleSerializer,
    'dill': DillSerializer,
    'torch': TorchSerializer,
    'torch::state': TorchStateSerializer,
}
