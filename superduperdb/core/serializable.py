import dataclasses as dc
import importlib
import inspect
import typing as t


def _deserialize(r, db=None):
    if isinstance(r, list):
        return [_deserialize(i, db=db) for i in r]

    if not isinstance(r, dict):
        return r

    if not ({'cls', 'dict', 'module'} <= set(r)):
        return {k: _deserialize(v, db=db) for k, v in r.items()}

    module = importlib.import_module(r['module'])
    component_cls = getattr(module, r['cls'])

    kwargs = _deserialize(r['dict'])
    if 'db' in inspect.signature(component_cls.__init__).parameters:
        kwargs.update(db=db)

    return component_cls(**kwargs)


def _serialize(item: t.Any) -> t.Dict[str, t.Any]:
    def fix(k, v):
        attr = getattr(item, k)
        if isinstance(attr, Serializable):
            return _serialize(attr)

        if isinstance(attr, list):
            for i, sc in enumerate(attr):
                if isinstance(sc, Serializable):
                    v[i] = _serialize(sc)

        return v

    d = {k: fix(k, v) for k, v in item.dict().items()}

    return {
        'cls': item.__class__.__name__,
        'dict': d,
        'module': item.__class__.__module__,
    }


@dc.dataclass
class Serializable:
    deserialize = staticmethod(_deserialize)
    serialize = _serialize

    def dict(self):
        return dc.asdict(self)
