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

    instance = component_cls(**{k: v for k, v in kwargs.items() if k != 'version'})

    # special handling of Component.version
    from superduperdb.components.component import Component

    if issubclass(component_cls, Component):
        instance.version = r.get('version', None)

    return instance


def _serialize(item: t.Any, serialize_variables: bool = True):
    if isinstance(item, Serializable):
        sub_dict = {
            k: _serialize(getattr(item, k), serialize_variables=serialize_variables)
            for k in item.dict()
        }
        out = {
            'cls': item.__class__.__name__,
            'dict': sub_dict,
            'module': item.__class__.__module__,
        }
        from superduperdb.components.component import Component

        if isinstance(item, Component):
            out.update(
                {
                    'type_id': item.type_id,
                    'identifier': item.identifier,
                    'version': getattr(
                        item,
                        'version',
                        None,
                    ),  # type: ignore[dict-item]
                }
            )
        return out
    if isinstance(item, dict):
        return {
            k: _serialize(v, serialize_variables=serialize_variables)
            for k, v in item.items()
        }
    if isinstance(item, (list, tuple)):
        return [_serialize(x, serialize_variables=serialize_variables) for x in item]
    if serialize_variables:
        if isinstance(item, Variable):
            return {
                'cls': 'Variable',
                'dict': {'value': item.value, 'setter_callback': item.setter_callback},
                'module': 'superduperdb.base.serializable',
            }
    return item


class VariableError(Exception):
    ...


def _find_variables(r):
    from .document import Document

    if isinstance(r, dict):
        return sum([_find_variables(v) for v in r.values()], [])
    elif isinstance(r, (list, tuple)):
        return sum([_find_variables(v) for v in r], [])
    elif isinstance(r, Variable):
        return [r]
    elif isinstance(r, Document):
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
        return [_replace_variables(v, db) for v in x]
    if isinstance(x, Variable):
        return x.set(db, **kwargs)
    if isinstance(x, Document):
        return x.set_variables(db, **kwargs)
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
    def variables(self) -> t.List['Variable']:
        out = {}
        v = _find_variables(self.dict())
        for var in v:
            out[var.value] = var
        return sorted(list(out.values()), key=lambda x: x.value)

    def set_variables(self, db, **kwargs) -> 'Serializable':
        r = self.serialize(
            serialize_variables=False
        )  # get serialized version of class instance
        r = _replace_variables(r, db, **kwargs)  # replace variables with values
        return self.deserialize(r)  # rebuild new class instance

    def dict(self) -> t.Dict[str, t.Any]:
        return asdict(self)


class Variable:
    """
    Mechanism for allowing "free variables" in a serializable object.
    The idea is to allow a variable to be set at runtime, rather than
    at object creation time.

    :param value: The name of the variable to be set at runtime.
    :param setter_callback: A callback function that takes the value, datalayer
                            and kwargs as input and returns the formatted
                            variable.
    """

    def __init__(
        self, value: t.Union[str, t.Any], setter_callback: t.Optional[t.Callable] = None
    ):
        self.value = value
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
