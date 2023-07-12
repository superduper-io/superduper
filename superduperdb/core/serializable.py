import dataclasses as dc
import importlib
import inspect
import typing as t


def _deserialize(r, db=None):
    if isinstance(r, dict) and {'cls', 'module', 'dict'}.issubset(set(r.keys())):
        module = importlib.import_module(f'{r["module"]}')
        component_cls = getattr(module, r['cls'])
        if 'db' in inspect.signature(component_cls.__init__).parameters:
            return component_cls(**_deserialize(r['dict']), db=db)
        else:
            return component_cls(**_deserialize(r['dict']))
    elif isinstance(r, dict):
        for k, v in r.items():
            r[k] = _deserialize(v, db=db)
    elif isinstance(r, list):
        for i, x in enumerate(r):
            r[i] = _deserialize(x, db=db)
    return r


def _serialize(item: t.Any) -> t.Dict[str, t.Any]:
    d = item.dict()
    for k in d:
        c = getattr(item, k)
        if isinstance(c, Serializable):
            d[k] = _serialize(c)
        elif isinstance(c, list):
            for i, sc in enumerate(c):
                if isinstance(sc, Serializable):
                    d[k][i] = _serialize(sc)
        # TODO: what about dict?
    return {
        'cls': item.__class__.__name__,
        'module': item.__class__.__module__,
        'dict': d,
    }


@dc.dataclass
class Serializable:
    deserialize = staticmethod(_deserialize)
    serialize = _serialize

    def dict(self):
        return dc.asdict(self)
