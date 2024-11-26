import importlib


def import_object(path):
    """Import item from path.

    :param path: Path to import from.
    """
    module = '.'.join(path.split('.')[:-1])
    cls = path.split('.')[-1]
    return getattr(importlib.import_module(module), cls)
