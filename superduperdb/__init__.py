from .base import config, configs, jsonable, logger
from .misc.superduper import superduper

__all__ = 'CFG', 'ICON', 'JSONable', 'ROOT', 'config', 'log', 'logging', 'superduper'

ICON = '🔮'
CFG = configs.CFG
JSONable = jsonable.JSONable
ROOT = configs.ROOT

logging = log = logger.logging

__version__ = '0.0.11'
