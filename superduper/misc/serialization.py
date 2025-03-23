import copy
import dataclasses as dc
import types
import typing as t

_ATOMIC_TYPES = frozenset(
    {
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


def asdict(obj, *, copy_method=copy.copy) -> t.Dict[str, t.Any]:
    """Convert the dataclass instance to a dict.

    Custom ``asdict`` function which exports a dataclass object into a dict,
    with a option to choose for nested non atomic objects copy strategy.

    :param obj: The dataclass instance to
    :param copy_method: The copy method to use for non atomic objects
    """
    if not dc.is_dataclass(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    return _asdict_inner(obj, dict, copy_method, top=True)


def _asdict_inner(obj, dict_factory, copy_method, top=False) -> t.Any:
    from superduper.base import Base
    from superduper.base.datatype import Saveable
    from superduper.misc.importing import isreallyinstance

    if type(obj) in _ATOMIC_TYPES:
        return obj
    elif not top and isreallyinstance(obj, Base):
        return obj
    elif not top and isreallyinstance(obj, Saveable):
        return obj
    elif dc.is_dataclass(obj) and not isinstance(obj, type):
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
