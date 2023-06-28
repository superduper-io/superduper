from functools import cached_property, wraps
from pydantic import BaseModel, Field
import abc
import sys
import typing as t

__all__ = 'Box', 'Factory', 'JSONable'

Contents = t.TypeVar('Contents')

TYPE_ID_ATTR = 'type_id'


def Factory(factory: t.Callable, **ka) -> t.Any:
    return Field(default_factory=factory, **ka)


class JSONable(BaseModel):
    """
    JSONable is the base class for all superduperdb classes that can be
    converted to and from JSON
    """

    class Config:
        # Fail in deserializion if there are extra fields
        extra = 'forbid'

        # See https://github.com/samuelcolvin/pydantic/issues/1241
        keep_untouched = (cached_property,)

    SUBCLASSES: t.ClassVar[t.Set[t.Type]] = set()
    TYPE_ID_TO_CLASS: t.ClassVar[t.Dict[str, t.Type]] = {}

    @wraps(BaseModel.dict)
    def dict(self, *a, **ka):
        d = super().dict(*a, **ka)
        properties = self.schema()['properties']
        return {k: v for k, v in d.items() if k in properties}

    def deepcopy(self) -> 'JSONable':
        return self.copy(deep=True)

    def __init_subclass__(cls, *a, **ka):
        super().__init_subclass__(*a, **ka)
        cls.SUBCLASSES.add(cls)

        try:
            type_id = getattr(cls, TYPE_ID_ATTR)
        except AttributeError:
            return

        if not isinstance(type_id, str):
            (type_id,) = type_id.__args__

        if old_model := cls.TYPE_ID_TO_CLASS.get(type_id):
            print(f'Duplicate type_id: old={old_model}, new={cls}', file=sys.stderr)

        cls.TYPE_ID_TO_CLASS[type_id] = cls


class Box(JSONable, t.Generic[Contents], abc.ABC):
    """
    A JSONable box for unJSONable contents.
    """

    def __call__(self):
        return self._contents

    @classmethod
    def box(cls, contents: Contents) -> 'Box[Contents]':
        box = cls._contents_to_box(contents)
        box.__dict__['_contents'] = contents
        return box

    @cached_property
    def _contents(self) -> Contents:
        return self._box_to_contents()

    def _box_to_contents(self) -> Contents:
        raise NotImplementedError

    @classmethod
    def _contents_to_box(cls, contents: Contents) -> 'Box[Contents]':
        raise NotImplementedError
