import importlib

from superduper import logging


def load_plugin(name: str):
    """Load a plugin by name.

    :param name: The name of the plugin to load.
    """
    if name == 'local':
        return importlib.import_module('superduper.backends.local')
    if name == 'ray':
        return importlib.import_module('superduper_services.compute.ray.compute')
    logging.info(f"Loading plugin: {name}")
    plugin = importlib.import_module(f'superduper_{name}')
    return plugin
