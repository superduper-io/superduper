import collections
import sys
import types
import typing as t
from abc import ABC
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any, ForwardRef, get_args, get_origin

from superduper import logging
from superduper.base.base import Base
from superduper.components.component import Component
from superduper.misc import typing as superduper_typing

ORDER = [
    'str',
    'int',
    'float',
    'bool',
    'json',
    'dill',
]


class Annotation(ABC):
    """Base class for all annotations."""

    def __init__(self):
        pass

    def __repr__(self):
        out = str(self.__class__.__name__)
        if hasattr(self, 'args') and self.args:
            out += '[' + ', '.join(str(arg) for arg in self.args) + ']'
        return out

    @staticmethod
    def build(annotation: t.Type):
        """Build an annotation from a type.

        :param annotation: The type to build the annotation from.
        """
        if isinstance(annotation, ForwardRef):
            module_globals = sys.modules[annotation.__module__].__dict__
            superduper_globals = sys.modules["superduper"].__dict__
            origin = _evaluate_forward_ref(
                annotation, {**module_globals, **superduper_globals}
            )
        else:
            origin = get_origin(annotation)

        if origin is None:
            return ANNOTATIONS[annotation]()

        args = get_args(annotation)
        return ANNOTATIONS[origin](*args)

    @property
    def args(self) -> t.List['Annotation']:
        """Return the arguments of the annotation."""
        return []

    @property
    def base_types(self) -> t.Set[str]:
        """Return the base type of the annotation."""
        out: t.Set = set()
        for arg in self.args:
            out = out | arg.base_types
        return out

    @property
    def datatype(self) -> str:
        """Return the datatype of the annotation."""
        bt = self.base_types
        if 'dill' in bt:
            return 'dill'
        if 'file' in bt:
            allowed = {
                'Path': 'file',
                'List[Path]': 'flist',
                'Dict[Str, Path]': 'fdict',
            }
        if 'componenttype' in bt:
            allowed = {
                'Component': 'componenttype',
                'List[Component]': 'componentlist',
                'Dict[Str, Component]': 'componentdict',
            }
            assert (
                str(self) in allowed
            ), f"Invalid component type: {str(self)}; supported: {allowed}"
            return allowed[str(self)]
        if 'basetype' in bt:
            assert (
                str(self) == 'Base'
            ), f"Invalid base type: {str(self)}; expected 'Base'"
            return 'basetype'
        return 'json'


class Literal(Annotation):
    """Annotation for Literal types.

    :param items: The items in the Literal.
    """

    def __init__(self, *items: t.Any):
        self.items = items

    @property
    def args(self):
        """Return the arguments of the annotation."""
        return []

    @property
    def base_types(self) -> t.Set[str]:
        """Return the base types of the annotation."""
        return {type(item).__name__ for item in self.items}

    @property
    def datatype(self) -> str:
        """Return the datatype of the annotation."""
        if self.base_types.issubset({'float', 'int', 'str', 'bool'}):
            return 'json'
        return 'dill'


class Union(Annotation):
    """Annotation for Union types.

    :param items: The items in the Union.
    """

    def __init__(self, *items: t.Type):
        self.items = [Annotation.build(arg) for arg in items if arg is not type(None)]

    @property
    def args(self):
        """Return the arguments of the annotation."""
        return self.items

    @property
    def datatype(self):
        """Return the datatype of the annotation."""
        if len(self.items) == 1:
            return self.items[0].datatype
        else:
            return super().datatype


class Dict(Annotation):
    """Annotation for Dict types.

    :param key_type: The type of the keys in the Dict.
    :param value_type: The type of the values in the Dict.
    """

    def __init__(
        self, key_type: t.Type | None = None, value_type: t.Type | None = None
    ):
        if key_type is None:
            key_type = str
        if value_type is None:
            value_type = str
        self.key_type = Annotation.build(key_type)
        self.value_type = Annotation.build(value_type)

    @property
    def args(self):
        return [self.key_type, self.value_type]


class Tuple(Annotation):
    """Annotation for Tuple types.

    :param items: The items in the Tuple.
    """

    def __init__(self, *items: t.Type):
        self.items = [Annotation.build(arg) for arg in items]

    @property
    def args(self):
        """Return the arguments of the annotation."""
        return self.items


class List(Annotation):
    """Annotation for List types.

    :param item_type: The type of the items in the List.
    """

    def __init__(self, item_type: t.Type | None = None):
        if item_type is None:
            item_type = str
        self.item_type = Annotation.build(item_type)

    @property
    def args(self):
        """Return the arguments of the annotation."""
        return [self.item_type]


class Str(Annotation):
    """Annotation for Str types."""

    @property
    def base_types(self) -> t.Set[str]:
        """Return the base types of the annotation."""
        return {'str'}

    @property
    def datatype(self) -> str:
        """Return the datatype of the annotation."""
        return 'str'


class Int(Annotation):
    """Annotation for Int types."""

    @property
    def base_types(self) -> t.Set[str]:
        """Return the base types of the annotation."""
        return {'int'}

    @property
    def datatype(self) -> str:
        """Return the datatype of the annotation."""
        return 'int'


class Float(Annotation):
    """Annotation for Float types."""

    @property
    def base_types(self) -> t.Set[str]:
        """Return the base types of the annotation."""
        return {'float'}

    @property
    def datatype(self) -> str:
        """Return the datatype of the annotation."""
        return 'float'


class Bool(Annotation):
    """Annotation for Bool types."""

    @property
    def base_types(self) -> t.Set[str]:
        """Return the base types of the annotation."""
        return {'bool'}

    @property
    def datatype(self) -> str:
        """Return the datatype of the annotation."""
        return 'bool'


class PathAnnotation(Annotation):
    """Annotation for Path types."""

    @property
    def base_types(self) -> t.Set[str]:
        """Return the base types of the annotation."""
        return {'file'}

    @property
    def datatype(self) -> str:
        """Return the datatype of the annotation."""
        return 'file'

    def __repr__(self):
        return 'Path'


class ComponentAnnotation(Annotation):
    """Annotation for Component types."""

    @property
    def base_types(self) -> t.Set[str]:
        """Return the base types of the annotation."""
        return {'componenttype'}

    def __repr__(self):
        return 'Component'

    @property
    def datatype(self) -> str:
        """Return the datatype of the annotation."""
        return 'componenttype'


class BaseAnnotation(Annotation):
    """Annotation for Base types."""

    @property
    def base_types(self) -> t.Set[str]:
        """Return the base types of the annotation."""
        return {'basetype'}

    @property
    def datatype(self) -> str:
        """Return the datatype of the annotation."""
        return 'basetype'


class Dill(Annotation):
    """Annotation for Dill types."""

    @property
    def base_types(self) -> t.Set[str]:
        """Return the base types of the annotation."""
        return {'dill'}

    @property
    def datatype(self) -> str:
        """Return the datatype of the annotation."""
        return 'dill'


class JSON(Annotation):
    """Annotation for JSON types."""

    @property
    def base_types(self) -> t.Set[str]:
        """Return the base types of the annotation."""
        return {'str', 'int', 'float', 'bool'}

    @property
    def datatype(self) -> str:
        """Return the datatype of the annotation."""
        return 'json'


class ArgumentDefaultDict(defaultdict):
    """A defaultdict that uses a factory function to create default values.

    # noqa
    """

    def __getitem__(self, key):
        if key not in self:
            item = self.default_factory(key)
            self[key] = item
        return super().__getitem__(key)


def default_factory(key):
    """Default factory function for the ArgumentDefaultDict.

    :param key: The key for which to create a default value.
    """
    try:
        if issubclass(key, Component):
            return ComponentAnnotation
        if issubclass(key, Base):
            return BaseAnnotation
    except TypeError as e:
        if 'arg 1 must be' in str(e):
            pass
        else:
            raise e
    return Dill


ANNOTATIONS = ArgumentDefaultDict(
    default_factory,  # type: ignore[arg-type]
    {
        t.Union: Union,
        types.UnionType: Union,
        dict: Dict,
        list: List,
        str: Str,
        tuple: Tuple,
        int: Int,
        float: Float,
        bool: Bool,
        Path: PathAnnotation,
        collections.abc.Sequence: List,
        t.Literal: Literal,
        superduper_typing.JSON: JSON,
        superduper_typing.File: PathAnnotation,
        superduper_typing.FileDict: lambda: Dict[Str, PathAnnotation],
        superduper_typing.SDict: lambda: Dict[Str, ComponentAnnotation],
        superduper_typing.SList: lambda: List[ComponentAnnotation],
        superduper_typing.BaseType: BaseAnnotation,
        superduper_typing.ComponentType: ComponentAnnotation,
    },
)


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
    if hasattr(cls, 'metadata_fields'):
        assert isinstance(
            cls.metadata_fields, dict
        ), f"Expected metadata_fields to be a dict, got {type(cls.metadata_fields)}"
        for field_name, field_type in cls.metadata_fields.items():
            if field_name not in hints:
                hints[field_name] = _safe_resolve_annotation(
                    field_type, {**module_globals, **superduper_globals, **mro_globals}
                )
    return hints


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

        mapped = Annotation.build(annotation)
        schema[parameter] = mapped.datatype

    return schema, annotations
