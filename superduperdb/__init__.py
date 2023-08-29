from .base import config, configs, jsonable, logger

__all__ = 'CFG', 'ICON', 'JSONable', 'ROOT', 'config', 'log', 'logging', 'superduper'

ICON = '🔮'
CFG = configs.CFG
JSONable = jsonable.JSONable
ROOT = configs.ROOT

from .misc.superduper import superduper  # noqa: E402

logging = log = logger.logging

__version__ = '0.0.6'
