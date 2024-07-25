import importlib

from superduper import logging


def load_plugin(name: str):
    if name == 'local':
        return importlib.import_module('superduper.backends.local')
    logging.info(f"Loading plugin: {name}")
    plugin = importlib.import_module(f'superduper_{name}')
    return plugin
