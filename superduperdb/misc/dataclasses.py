import functools
import typing as t
import warnings
from dataclasses import (
    MISSING,
    InitVar,
    asdict,
    astuple,
    field,
    fields,
    is_dataclass,
    replace,
)

from pydantic import dataclasses as pdc

__all__ = (
    'InitVar',
    'MISSING',
    'add_methods',
    'asdict',
    'astuple',
    'field',
    'fields',
    'is_dataclass',
    'replace',
)

_METHODS = 'asdict', 'astuple', 'dict', 'fields', 'replace'
_CLASS_METHODS = ('fields',)


@functools.wraps(pdc.dataclass)
def dataclass(cls=None, **kwargs):
    if not cls:
        return functools.partial(dataclass, **kwargs)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        return add_methods(pdc.dataclass(cls, **kwargs))


def add_methods(dcls: t.Any) -> t.Any:
    """Adds dataclasses functions as methods to a dataclass.

    Adds four new instance methods, `asdict()`, `astuple()`, `dict()`, and `replace()`
    and a new class method, `fields()`.
    """
    for m in _METHODS:
        if not hasattr(dcls, m):
            method = asdict if m == 'dict' else globals()[m]
            if m in _CLASS_METHODS:
                method = classmethod(method)
            setattr(dcls, m, method)

    return dcls
