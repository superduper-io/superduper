import sys
import traceback
import types
import typing as t
from dataclasses import fields
from typing import Any, ForwardRef, get_args, get_origin

from superduper import logging
from superduper.base.base import Base
from superduper.components.component import Component


def gather_mro_globals(cls):
    """Return a merged dictionary of the module global from the MRO of `cls`.

    :param cls: The class to gather the MRO globals from.
    """
    merged = {}
    for base in cls.__mro__:
        # We skip anything that doesn't have a __module__ (e.g., a built-in type)
        if hasattr(base, "__module__") and base.__module__ in sys.modules:
            merged.update(sys.modules[base.__module__].__dict__)
    return merged


def _evaluate_forward_ref(ref: ForwardRef, globalns: dict, localns: dict = None):
    # Call ref._evaluate(...) with the correct signature depending on Python version.
    return ref._evaluate(globalns, localns or {}, set())  # type: ignore[arg-type]


def _safe_resolve_annotation(raw_annotation: Any, globalns: dict) -> Any:
    # Attempt to resolve a single annotation (which may be a string forward reference,
    # a ForwardRef object, or already a real Python type). If it can't be resolved
    # because the referenced name doesn't exist, return None (or any placeholder).

    # 1) If annotation is already a real type (e.g. int, list[str], etc.),
    # just return it
    if not isinstance(raw_annotation, str) and not isinstance(
        raw_annotation, ForwardRef
    ):
        return raw_annotation

    # 2) If annotation is a string, convert it to a ForwardRef object
    if isinstance(raw_annotation, str):
        raw_annotation_eval = ForwardRef(raw_annotation)
    else:
        raw_annotation_eval = raw_annotation

    # 3) Attempt to _evaluate the forward reference
    try:
        out = _evaluate_forward_ref(raw_annotation_eval, globalns, localns={})
    except NameError:
        # The name in the annotation doesn't exist -> skip or fallback
        return None

    return out


def _safe_get_type_hints(cls: t.Type[t.Any]) -> dict[str, t.Any]:
    # Partially resolve type hints for a dataclass `cls`. For each field:
    #   - If its annotation can be resolved, return the real type.
    #   - If not, return None instead of raising NameError.
    # We'll pull the module in which `cls` is defined, so that forward references
    # can see the same globals that `cls` sees:
    mro_globals = gather_mro_globals(cls)
    module_globals = sys.modules[cls.__module__].__dict__
    superduper_globals = sys.modules["superduper"].__dict__

    hints = {}

    for f in fields(cls):
        hints[f.name] = _safe_resolve_annotation(
            f.type, {**module_globals, **superduper_globals, **mro_globals}
        )
    return hints


def _process_dict(args):
    if not args:
        return dict, None

    assert len(args) == 2
    # only support strings to our things
    if args[0] is str:
        return args[1], dict


def _process_list(args):
    if len(args) == 0:
        return list, None
    assert len(args) == 1
    return args[0], list


def _process_union(args):
    if len(args) == 1:
        return process(args[0])

    elif len(args) == 2 and args[-1] is type(None):
        return process(args[0])

    elif len(args) > 2 and args[-1] is type(None):
        return process(args[:-1])

    else:
        # arbitrary union not supported
        return None, None


def _process_literal(args):

    # Literal is a special case, we need to get the first argument
    # and check if it's a string or not
    if all(isinstance(t, str) for t in args):
        inferred_cls = str
    elif all(isinstance(t, int) for t in args):
        inferred_cls = int
    else:
        inferred_cls = None
    iterable_ = None
    return inferred_cls, iterable_


class _DataTypeFactory:
    def __init__(self, source, name):
        self.source = source
        self.name = name

    def __getitem__(self, cls):
        if cls in {str, int, bool, float}:
            return cls.__name__
        if cls in {list, dict}:
            return "json"
        try:
            if isinstance(cls, t.NewType):
                return str(cls).split(".")[-1].lower()
            if issubclass(cls, Component):
                return "componenttype"
            if issubclass(cls, Base):
                return "basetype"
        except TypeError:
            pass
        return "dill"


def _map_type_to_superduper(source, name, cls, iterable):
    if iterable is None:
        return _DataTypeFactory(source, name)[cls]
    if cls and issubclass(cls, Base):
        if iterable is list:
            return "componentlist"
        if iterable is dict:
            return "componentdict"
        raise ValueError(f"Unsupported iterable type {iterable} for {cls}")
    if cls is None and iterable in {list, dict}:
        return "json"
    if cls in {str, int, float, bool, dict}:
        return "json"
    return "dill"


def process(annotation):
    """Process an annotation with a crude mapping to workable superduper types.

    Output is expected as a tuple of base type and iterable over that type.

    :param annotation: The annotation to process.

    >>> import typing as t
    >>> from superduper import Model, Component
    >>> process(Model)
    (superduper.components.model.Model, None)
    >>> process(Model | None)
    (superduper.components.model.Model, None)
    >>> process(t.List[str])
    (str, list)
    >>> process(t.Dict[str, Component])
    (superduper.components.component.Component, dict)
    >>> process(t.Dict[str, MyClass])
    (my_path.to_module.MyClass, dict)
    """
    origin = get_origin(annotation)
    args = get_args(annotation)

    inferred_cls = None
    iterable_ = None

    if origin is None:
        inferred_cls = annotation

    if origin is t.Union or origin is types.UnionType:
        inferred_cls, iterable_ = _process_union(args)

    if origin is list:
        inferred_cls, iterable_ = _process_list(args)

    if origin is dict:
        inferred_cls, iterable_ = _process_dict(args)

    if origin is t.Literal:
        inferred_cls, iterable_ = _process_literal(args)

    if isinstance(inferred_cls, ForwardRef):
        module_globals = sys.modules[inferred_cls.__module__].__dict__
        superduper_globals = sys.modules["superduper"].__dict__
        inferred_cls = _evaluate_forward_ref(
            inferred_cls, {**module_globals, **superduper_globals}
        )

    return inferred_cls, iterable_


def get_schema(cls):
    """Get a schema for a superduper class.

    :param cls: The class to get a schema for.
    """
    annotations = _safe_get_type_hints(cls)

    schema = {}
    for parameter in annotations:
        annotation = annotations[parameter]
        if annotation is None:
            schema[parameter] = "str"
            continue
        inferred_cls, iterable = process(annotation)
        if inferred_cls is None:
            schema[parameter] = "dill"
            continue
        try:
            schema[parameter] = _map_type_to_superduper(
                cls.__name__, parameter, inferred_cls, iterable
            )
        except TypeError as e:
            logging.error(
                f"Error processing annotation {cls.__name__}.{parameter}: {annotation}"
            )
            raise e

    return schema, annotations
