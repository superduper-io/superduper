import functools
import sys
import typing as t

import pydantic

__all__ = 'Factory', 'JSONable'

TYPE_ID_ATTR = 'type_id'
_NONE = object()


def Factory(factory: t.Callable, **ka) -> t.Any:
    return pydantic.Field(default_factory=factory, **ka)


class JSONable(pydantic.BaseModel):
    """
    JSONable is the base class for all superduperdb classes that can be
    converted to and from JSON
    """

    class Config:
        # Fail in deserializion if there are extra fields
        extra = 'forbid'

        # See https://github.com/samuelcolvin/pydantic/issues/1241
        if pydantic.__version__.startswith('2'):
            ignored_types = (functools.cached_property,)
        else:
            keep_untouched = (functools.cached_property,)

    SUBCLASSES: t.ClassVar[t.Set[t.Type]] = set()
    TYPE_ID_TO_CLASS: t.ClassVar[t.Dict[str, t.Type]] = {}

    @functools.wraps(pydantic.BaseModel.dict)
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
