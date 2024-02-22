import dataclasses as dc
import importlib
import typing as t
from copy import deepcopy

from superduperdb.base.leaf import Leaf
from superduperdb.misc.serialization import asdict


def _from_dict(r: t.Any, db: None = None) -> t.Any:
    from superduperdb.base.document import Document
    from superduperdb.components.datatype import File, LazyArtifact

    if isinstance(r, Document):
        r = r.unpack(db, leaves_to_keep=(LazyArtifact, File))
    if isinstance(r, (list, tuple)):
        return [_from_dict(i, db=db) for i in r]
    if not isinstance(r, dict):
        return r
    if '_content' in r:
        r = r['_content']
    if 'cls' in r and 'module' in r and 'dict' in r:
        module = importlib.import_module(r['module'])
        cls_ = getattr(module, r['cls'])
        kwargs = _from_dict(r['dict'])
        kwargs_init = {k: v for k, v in kwargs.items() if k not in cls_.set_post_init}
        kwargs_post_init = {k: v for k, v in kwargs.items() if k in cls_.set_post_init}
        instance = cls_(**kwargs_init)
        for k, v in kwargs_post_init.items():
            setattr(instance, k, v)
        return instance
    else:
        return {k: _from_dict(v, db=db) for k, v in r.items()}


class VariableError(Exception):
    ...


def _find_variables(r):
    if isinstance(r, dict):
        return sum([_find_variables(v) for v in r.values()], [])
    elif isinstance(r, (list, tuple)):
        return sum([_find_variables(v) for v in r], [])
    elif isinstance(r, Variable):
        return [r]
    return []


def _replace_variables(x, db, **kwargs):
    from .document import Document

    if isinstance(x, dict):
        return {
            _replace_variables(k, db, **kwargs): _replace_variables(v, db, **kwargs)
            for k, v in x.items()
        }
    if isinstance(x, (list, tuple)):
        return [_replace_variables(v, db) for v in x]
    if isinstance(x, Variable):
        return x.set(db, **kwargs)
    if isinstance(x, Document):
        return x.set_variables(db, **kwargs)
    return x


@dc.dataclass
class Serializable(Leaf):
    """
    Base class for serializable objects. This class is used to serialize and
    deserialize objects to and from JSON + Artifact instances.
    """

    set_post_init: t.ClassVar[t.Sequence] = ()

    @property
    def unique_id(self):
        return str(hash(self.dict().encode()))

    @property
    def variables(self) -> t.List['Variable']:
        out = {}
        r = self.encode(leaf_types_to_keep=(Variable,))
        v = _find_variables(r)
        for var in v:
            out[var.value] = var
        return sorted(list(out.values()), key=lambda x: x.value)

    def set_variables(self, db, **kwargs) -> 'Serializable':
        """
        Set free variables of self.

        :param db:
        """
        r = self.encode(leaf_types_to_keep=(Variable,))
        r = _replace_variables(r, db, **kwargs)
        return self.decode(r)

    def encode(
        self,
        leaf_types_to_keep: t.Sequence = (),
    ):
        r = dict(self.dict().encode(leaf_types_to_keep=leaf_types_to_keep))
        r['leaf_type'] = 'serializable'
        return {'_content': r}

    @classmethod
    def decode(cls, r, db: t.Optional[t.Any] = None):
        return _from_dict(r, db=db)

    def dict(self):
        from superduperdb import Document

        return Document(asdict(self))

    def copy(self):
        return deepcopy(self)


@dc.dataclass
class Variable(Serializable):
    """
    Mechanism for allowing "free variables" in a serializable object.
    The idea is to allow a variable to be set at runtime, rather than
    at object creation time.

    :param value: The name of the variable to be set at runtime.
    :param setter_callback: A callback function that takes the value, datalayer
                            and kwargs as input and returns the formatted
                            variable.
    """

    value: t.Any
    setter_callback: dc.InitVar[t.Optional[t.Callable]] = None

    def __post_init__(self, setter_callback):
        self.setter_callback = setter_callback

    def __repr__(self) -> str:
        return '$' + str(self.value)

    def __hash__(self) -> int:
        return hash(self.value)

    def set(self, db, **kwargs):
        """
        Get the intended value from the values of the global variables.

        >>> Variable('number').set(db, number=1.5, other='test')
        1.5

        :param db: The datalayer instance.
        :param kwargs: Variables to be used in the setter_callback
                       or as formatting variables.
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
            return kwargs[self.value]
