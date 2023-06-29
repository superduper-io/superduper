import importlib


def to_dict(query):
    return {
        'module': query.__class__.__module__,
        'cls': query.__class__.__name__,
        'dict': query.dict(),
    }


def from_dict(r):
    module = importlib.import_module(f'{r["module"]}')
    cls = getattr(module, r['cls'])
    return cls(**r['dict'])
