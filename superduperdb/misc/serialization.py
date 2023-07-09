import importlib
import inspect
import io
import pickle
from abc import ABC

import dill
import torch


def to_dict(item):
    return {
        'module': item.__class__.__module__,
        'cls': item.__class__.__name__,
        'dict': item.dict(),
    }


def from_dict(r, db=None):
    module = importlib.import_module(f'{r["module"]}')
    cls = getattr(module, r['cls'])
    if 'db' in inspect.signature(cls.__init__).parameters:
        return cls(**r['dict'], db=db)
    else:
        return cls(**r['dict'])


class _BaseSerializer(ABC):
    @staticmethod
    def encode(self, object):
        raise NotImplementedError

    @staticmethod
    def decode(self, bytes):
        raise NotImplementedError


class PickleSerializer:
    @staticmethod
    def encode(object):
        f = io.BytesIO()
        pickle.dump(object, f)
        return f.getvalue()

    @staticmethod
    def decode(bytes):
        return pickle.load(io.BytesIO(bytes))


class DillSerializer:
    @staticmethod
    def encode(object):
        f = io.BytesIO()
        dill.dump(object, f)
        return f.getvalue()

    @staticmethod
    def decode(bytes):
        return dill.load(io.BytesIO(bytes))


class TorchSerializer:
    @staticmethod
    def encode(object):
        was_gpu = object.device == 'cuda'
        object.to('cpu')
        f = io.BytesIO()
        object.save(f)
        if was_gpu:
            object.to('cuda')
        return f.getvalue()

    @staticmethod
    def decode(bytes):
        return torch.load(io.BytesIO(bytes))


class TorchStateSerializer:
    @staticmethod
    def encode(object, info):
        f = io.BytesIO()
        torch.save(object.state_dict(), f)
        return f.getvalue()

    @staticmethod
    def decode(object, info):
        instance = from_dict(info)
        instance.load_state_dict(object)
        return instance


serializers = {
    'pickle': PickleSerializer,
    'dill': DillSerializer,
    'torch': TorchSerializer,
    'torch::state': TorchStateSerializer,
}
