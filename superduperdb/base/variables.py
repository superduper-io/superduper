import dataclasses as dc
import typing as t

from superduperdb.base.leaf import Leaf
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
        return [r]
    if isinstance(r, Leaf):
        return r.variables
    return []


def _replace_variables(x, db, **kwargs):
    from .document import Document

    if isinstance(x, dict):
        return {
            _replace_variables(k, db, **kwargs): _replace_variables(v, db, **kwargs)
            for k, v in x.items()
        }
    if isinstance(x, (list, tuple)):
        return [_replace_variables(v, db, **kwargs) for v in x]
    if isinstance(x, Variable):
        return x.set(db, **kwargs)
    if isinstance(x, Document):
        return x.set_variables(db, **kwargs)
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

    def __post_init__(self, db, artifacts):
        super().__post_init__(db)
        self.value = self.identifier

    def __repr__(self) -> str:
        return f'${str(self.value)}'

    def __hash__(self) -> int:
        return hash(self.value)

    def set(self, db, **kwargs):
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
