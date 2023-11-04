import copy
import dataclasses as dc
import io
import pickle
import types
import typing as t
from abc import ABC

import dill
import typing_extensions as te

Info = t.Optional[t.Dict[str, t.Any]]

_ATOMIC_TYPES = frozenset(
    {
        # Common JSON Serializable types
        type(None),
        bool,
        int,
        float,
        str,
        # Other common types
        complex,
        bytes,
        # Other types that are also unaffected by deepcopy
        type(...),
        type(NotImplemented),
        types.CodeType,
        types.BuiltinFunctionType,
        types.FunctionType,
        type,
        range,
        property,
    }
)


class ModuleClassDict(te.TypedDict):
    """
    A ``dict``with the module, class and JSONization of an object

    :param module: The module of the object
    """

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
        import torch

        from superduperdb.ext.torch.utils import device_of

        if not isinstance(object, dict):
            previous_device = str(device_of(object))
            object.to('cpu')
            f = io.BytesIO()
            torch.save(object, f)
            object.to(previous_device)
        else:
            f = io.BytesIO()
            torch.save(object, f)
        return f.getvalue()

    @staticmethod
    def decode(b: bytes, info: Info = None) -> t.Any:
        import torch

        return torch.load(io.BytesIO(b))


class Serializers:
    serializers: t.Dict[str, t.Type] = {}

    def add(self, name: str, serializer: t.Type):
        self.serializers[name] = serializer

    def __getitem__(self, serializer):
        return self.serializers[serializer]


serializers = Serializers()
serializers.add('pickle', PickleSerializer)
serializers.add('dill', DillSerializer)
serializers.add('torch', TorchSerializer)


class Method:
    """
    A callable that calls a method on the object it is called with.

    :param method: The method to call.
    :param *args: The args to call the method with.
    :param **kwargs: The kwargs to call the method with.
    """

    def __init__(self, method: str, *args, **kwargs):
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def __call__(self, X: t.Any) -> t.Any:
        return getattr(X, self.method)(*self.args, **self.kwargs)


def asdict(obj, *, copy_method=copy.copy) -> t.Dict[str, t.Any]:
    """
    Custom ``asdict`` function which exports a dataclass object into a dict,
    with a option to choose for nested non atomic objects copy strategy.
    """
    if not dc.is_dataclass(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    return _asdict_inner(obj, dict, copy_method)


def _asdict_inner(obj, dict_factory, copy_method) -> t.Any:
    if type(obj) in _ATOMIC_TYPES:
        return obj
    elif dc.is_dataclass(obj):
        # fast path for the common case
        return {
            f.name: _asdict_inner(getattr(obj, f.name), dict, copy_method)
            for f in dc.fields(obj)
        }
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        # obj is a namedtuple.  Recurse into it, but the returned
        # object is another namedtuple of the same type.  This is
        # similar to how other list- or tuple-derived classes are
        # treated (see below), but we just need to create them
        # differently because a namedtuple's __init__ needs to be
        # called differently (see bpo-34363).

        # I'm not using namedtuple's _asdict()
        # method, because:
        # - it does not recurse in to the namedtuple fields and
        #   convert them to dicts (using dict_factory).
        # - I don't actually want to return a dict here.  The main
        #   use case here is json.dumps, and it handles converting
        #   namedtuples to lists.  Admittedly we're losing some
        #   information here when we produce a json list instead of a
        #   dict.  Note that if we returned dicts here instead of
        #   namedtuples, we could no longer call asdict() on a data
        #   structure where a namedtuple was used as a dict key.

        return type(obj)(*[_asdict_inner(v, dict_factory, copy_method) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_asdict_inner(v, dict_factory, copy_method) for v in obj)
    elif isinstance(obj, dict):
        if hasattr(type(obj), 'default_factory'):
            # obj is a defaultdict, which has a different constructor from
            # dict as it requires the default_factory as its first arg.
            result = type(obj)(getattr(obj, 'default_factory'))
            for k, v in obj.items():
                result[_asdict_inner(k, dict_factory, copy_method)] = _asdict_inner(
                    v, dict_factory, copy_method
                )
            return result
        return type(obj)(
            (
                _asdict_inner(k, dict_factory, copy_method),
                _asdict_inner(v, dict_factory, copy_method),
            )
            for k, v in obj.items()
        )
    else:
        return copy_method(obj)
