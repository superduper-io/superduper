import dataclasses as dc
import re
import typing as t

from superduperdb.base.leaf import Leaf
from superduperdb.components.schema import Schema
from superduperdb.misc.annotations import merge_docstrings


class VariableError(Exception):
    """
    Variable error.

    :param args: *args for `Exception`.
    :param kwargs: **kwargs for `Exception`.
    """


def _find_variables(r):
    if isinstance(r, dict):
        return sum([_find_variables(v) for v in r.values()], [])
    if isinstance(r, (list, tuple)):
        return sum([_find_variables(v) for v in r], [])
    if isinstance(r, Variable):
        return [r.identifier]
    if isinstance(r, str):
        return re.findall(r'<var:(.*?)>', r)
    if isinstance(r, Leaf):
        return r.variables
    return []


def _find_variables_with_path(r):
    if isinstance(r, dict):
        out = []
        for k, v in r.items():
            tmp = _find_variables_with_path(v)
            for p in tmp:
                out.append({'path': [k] + p['path'], 'variable': p['variable']})
        return out
    elif isinstance(r, (list, tuple)):
        out = []
        for i, v in enumerate(r):
            tmp = _find_variables_with_path(v)
            for p in tmp:
                out.append({'path': [i] + p['path'], 'variable': p['variable']})
        return out
    elif isinstance(r, Variable):
        return [{'path': [], 'variable': r}]
    return []


def _replace_variables(x, **kwargs):
    from .document import Document

    if isinstance(x, dict):
        return {
            _replace_variables(k, **kwargs): _replace_variables(v, **kwargs)
            for k, v in x.items()
        }
    if isinstance(x, str):
        variables = re.findall(r'<var:(.*?)>', x)
        variables = list(map(lambda v: v.strip(), variables))
        for k, v in kwargs.items():
            if k in variables:
                x = x.replace(f'<var:{k}>', str(v))

        return x
    if isinstance(x, (list, tuple)):
        return [_replace_variables(v, **kwargs) for v in x]
    if isinstance(x, Variable):
        return x.set(**kwargs)
    if isinstance(x, Document):
        return x.set_variables(**kwargs)
    return x


@merge_docstrings
@dc.dataclass
class Variable(Leaf):
    """Mechanism for allowing "free variables" in a leaf object.

    The idea is to allow a variable to be set at runtime, rather than
    at object creation time.
    """

    def __post_init__(self, db=None):
        super().__post_init__(db)
        self.value = self.identifier

    @property
    def _id(self):
        return f'variable/{self.identifier}'

    @property
    def key(self):
        """Variable key."""
        return f'<var:{str(self.value)}>'

    def _deep_flat_encode(
        self, cache, blobs, files, leaves_to_keep=(), schema: t.Optional[Schema] = None
    ):
        r = super()._deep_flat_encode(cache, blobs, files, leaves_to_keep, schema)
        if isinstance(self, leaves_to_keep):
            cache[self._id] = self
            return f'?{self._id}'
        cache[self._id] = r
        return f'?{self._id}'

    def __repr__(self) -> str:
        return self.key

    def __hash__(self) -> int:
        return hash(self.value)

    def set(self, **kwargs):
        """
        Get the intended value from the values of the global variables.

        :param db: The datalayer instance.
        :param kwargs: Variables to be used in the setter_callback
                       or as formatting variables.

        >>> Variable('number').set(db, number=1.5, other='test')
        1.5

        """
        assert isinstance(self.value, str)
        return kwargs.get(self.value, self)
