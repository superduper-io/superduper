import dataclasses as dc
import typing as t

from superduperdb.base.leaf import Leaf
from superduperdb.misc.annotations import merge_docstrings

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.schema import Schema


class VariableError(Exception):
    """
    Variable error.

    :param args: *args for `Exception`.
    :param kwargs: **kwargs for `Exception`.
    """


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


def _replace_variables(x, db: t.Optional['Datalayer'] = None, **kwargs):
    from .document import Document

    if isinstance(x, dict):
        return {
            _replace_variables(k, db=db, **kwargs): _replace_variables(v, db=db, **kwargs)
            for k, v in x.items()
        }
    if isinstance(x, (list, tuple)):
        return [_replace_variables(v, db=db, **kwargs) for v in x]
    if isinstance(x, str) and x.startswith('?'):
        import re
        variables = re.findall(r'\?(\w+)', x)
        for v in variables:
            x = x.replace(f'?{v}', str(kwargs[v]))
        return x
    if isinstance(x, Variable):
        return x.set_variables(db=db, **kwargs)
    if isinstance(x, Document):
        return x.set_variables(db=db, **kwargs)
    return x


@merge_docstrings
@dc.dataclass
class Variable(Leaf):
    """Mechanism for allowing "free variables" in a leaf object.

    The idea is to allow a variable to be set at runtime, rather than
    at object creation time.

    :param setter_callback: A callback function that takes the value, datalayer
                            and kwargs as input and returns the formatted
                            variable.
    """

    setter_callback: dc.InitVar[t.Optional[t.Callable]] = None

    @property
    def _id(self):
        return f'variable/{self.identifier}'

    def __post_init__(self, db, artifacts):
        super().__post_init__(db)
        self.value = self.identifier

    def __repr__(self) -> str:
        return f'?{str(self.value)}'

    def __hash__(self) -> int:
        return hash(self.value)

    def _deep_flat_encode(self, cache, blobs, files, leaves_to_keep=(), schema: t.Optional['Schema'] = None):
        if isinstance(self, leaves_to_keep):
            cache[self._id] = self
            return f'@{self._id}'
        r = super()._deep_flat_encode(cache, blobs, files, leaves_to_keep, schema)
        cache[self._id] = r
        return f'@{self._id}'

    def set_variables(self, db: t.Optional['Datalayer'] = None, **kwargs):
        """
        Get the intended value from the values of the global variables.

        :param db: The datalayer instance.
        :param kwargs: Variables to be used in the setter_callback
                       or as formatting variables.

        >>> Variable('number').set(db, number=1.5, other='test')
        1.5

        """
        if self.setter_callback is not None:
            try:
                return self.setter_callback(db, self.value, kwargs)
            except Exception as e:
                raise VariableError(
                    f'Could not set variable {self.value} '
                    f'based on {self.setter_callback} and **kwargs: {kwargs}'
                ) from e
        else:
            assert isinstance(self.value, str)
            return kwargs.get(self.value, self)
