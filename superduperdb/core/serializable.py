import dataclasses as dc

import importlib
import inspect


@dc.dataclass
class Serializable:
    @classmethod
    def from_dict(cls, r, db=None):
        if isinstance(r, dict) and {'cls', 'module', 'dict'}.issubset(set(r.keys())):
            module = importlib.import_module(f'{r["module"]}')
            component_cls = getattr(module, r['cls'])
            if 'db' in inspect.signature(component_cls.__init__).parameters:
                return component_cls(**cls.from_dict(r['dict']), db=db)
            else:
                return component_cls(**cls.from_dict(r['dict']))
        elif isinstance(r, dict):
            for k, v in r.items():
                r[k] = cls.from_dict(v, db=db)
        elif isinstance(r, list):
            for i, x in enumerate(r):
                r[i] = cls.from_dict(x, db=db)
        return r

    def dict(self):
        return dc.asdict(self)

    def to_dict(self):
        d = self.dict()
        for k in d:
            c = getattr(self, k)
            if isinstance(c, Serializable):
                c = getattr(self, k)
                d[k] = c.to_dict()
            elif isinstance(c, list):
                for i, sc in enumerate(c):
                    if isinstance(sc, Serializable):
                        d[k][i] = sc.to_dict()
        return {
            'cls': self.__class__.__name__,
            'module': self.__class__.__module__,
            'dict': d,
        }
