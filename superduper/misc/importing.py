import importlib

from superduper import logging


def load_plugin(name: str):
    """Load a plugin by name.

    :param name: The name of the plugin to load.
    """
    if name in {'local', 'inmemory'}:
        return importlib.import_module('superduper.backends.{}'.format(name))
    logging.info(f"Loading plugin: {name}")
    plugin = importlib.import_module(f'superduper_{name}')
    return plugin


def import_object(path):
    """Import item from path.

    :param path: Path to import from.
    """
    module = '.'.join(path.split('.')[:-1])
    cls = path.split('.')[-1]
    return getattr(importlib.import_module(module), cls)


def isreallyinstance(this, cls):
    """Check if the component is an instance of a class.

    :param this: The component to check.
    :param cls: The class to check.
    """
    # no idea why this is sometimes necessary - may be IPython autoreload related
    mro = [f'{o.__module__}.{o.__name__}' for o in this.__class__.__mro__]
    return isinstance(this, cls) or f'{cls.__module__}.{cls.__name__}' in mro


def isreallysubclass(this, cls):
    """Check if the component is an instance of a class.

    :param this: The component to check.
    :param cls: The class to check.
    """
    # no idea why this is sometimes necessary - may be IPython autoreload related
    mro = [f'{o.__module__}.{o.__name__}' for o in this.__mro__]
    return issubclass(this, cls) or f'{cls.__module__}.{cls.__name__}' in mro
