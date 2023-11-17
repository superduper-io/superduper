import dataclasses as dc
import importlib
import inspect
import typing as t

from superduperdb.misc.serialization import asdict


def is_component_metadata(r: dict) -> bool:
    COMPONENT_KEYS = {'type_id', 'identifier', 'version'}
    if COMPONENT_KEYS == set(r):
        return True
    return False


def is_component(r: dict) -> bool:
    COMPONENT_KEYS = {'cls', 'dict', 'module'}
    if COMPONENT_KEYS <= set(r):
        return True
    return False


def _deserialize(r: t.Any, db: None = None) -> t.Any:
    if isinstance(r, (list, tuple)):
        return [_deserialize(i, db=db) for i in r]

    if not isinstance(r, dict):
        return r

    if not is_component(r):
        return {k: _deserialize(v, db=db) for k, v in r.items()}

    module = importlib.import_module(r['module'])
    component_cls = getattr(module, r['cls'])

    kwargs = _deserialize(r['dict'])
    if 'db' in inspect.signature(component_cls.__init__).parameters:
        kwargs.update(db=db)

    return component_cls(**kwargs)


def _serialize(item: t.Any) -> t.Dict[str, t.Any]:
    def unpack(k, v):
        attr = getattr(item, k)
        if isinstance(attr, Serializable):
            return _serialize(attr)

        if isinstance(attr, (list, tuple)):
            if isinstance(attr, tuple):
                v = list(v)

            for i, sc in enumerate(attr):
                if isinstance(sc, Serializable):
                    v[i] = _serialize(sc)

        if isinstance(attr, dict):
            for key, value in attr.items():
                if isinstance(value, Serializable):
                    v[key] = _serialize(value)
        return v

    d = {k: unpack(k, v) for k, v in item.dict().items()}

    from superduperdb.components.component import Component

    to_add = {}
    if isinstance(item, Component):
        to_add = {
            'type_id': item.type_id,
            'identifier': item.identifier,
            'version': getattr(item, 'version', None),
        }

    return {
        'cls': item.__class__.__name__,
        'dict': d,
        'module': item.__class__.__module__,
        **to_add,
    }


class VariableError(Exception):
    ...


class Variable:
    """
    Mechanism for allowing "free variables" in a serializable object.
    The idea is to allow a variable to be set at runtime, rather than
    at object creation time.

    :param value: The name of the variable to be set at runtime.
    """

    def __init__(self, value, setter_callback):
        self.value = value
        self.setter_callback = setter_callback

    def __repr__(self) -> str:
        return '$' + str(self.value)

    def __hash__(self) -> int:
        return hash(self.value)

    def set(self, db):
        """
        Get the intended value from the values of the global variables.

        >>> Variable('number').set(number=1.5, other='test')
        1.5

        :param db: The datalayer instance.
        """
        try:
            return self.setter_callback(db, self.value)
        except Exception as e:
            raise VariableError(
                f'Could not set variable {self.value} based on {self.setter_callback}'
            ) from e


def _find_variables(r):
    if isinstance(r, dict):
        return sum([_find_variables(v) for v in r.values()], [])
    elif isinstance(r, (list, tuple)):
        return sum([_find_variables(v) for v in r], [])
    elif isinstance(r, Variable):
        return [r]
    return []


def _replace_variables(x, db):
    if isinstance(x, dict):
        return {
            _replace_variables(k, db): _replace_variables(v, db) for k, v in x.items()
        }
    if isinstance(x, (list, tuple)):
        return [_replace_variables(v, db) for v in x]
    if isinstance(x, Variable):
        return x.set(db)
    return x


@dc.dataclass
class Serializable:
    """
    Base class for serializable objects. This class is used to serialize and
    deserialize objects to and from JSON + Artifact instances.
    """

    deserialize = staticmethod(_deserialize)
    serialize = _serialize

    @property
    def variables(self) -> t.List[Variable]:
        return _find_variables(self.dict())

    def set_variables(self, db) -> 'Serializable':
        r = self.serialize()  # get serialized version of class instance
        r = _replace_variables(r, db)  # replace variables with values
        return self.deserialize(r)  # rebuild new class instance

    def dict(self) -> t.Dict[str, t.Any]:
        return asdict(self)
