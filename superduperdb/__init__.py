from .misc import config, configs
from .misc.jsonable import JSONable

__all__ = 'CFG', 'ICON', 'JSONable', 'ROOT', 'config', 'log', 'logging'

CFG = configs.build_config()
ICON = '🔮'
ROOT = configs.ROOT

from .misc import logger  # noqa: E402

log = logger.logging
