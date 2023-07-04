import importlib


def to_dict(item):
    return {
        'module': item.__class__.__module__,
        'cls': item.__class__.__name__,
        'dict': item.dict(),
    }


def from_dict(r):
    module = importlib.import_module(f'{r["module"]}')
    cls = getattr(module, r['cls'])
    return cls(**r['dict'])
