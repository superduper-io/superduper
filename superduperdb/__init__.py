from .misc import config, configs
from .misc.jsonable import JSONable
from pathlib import Path
import os

__all__ = 'CFG', 'ICON', 'JSONable', 'ROOT', 'config'

CFG = configs.CONFIG.config
ICON = '🔮'
ROOT = Path(__file__).parent

if openai := CFG.apis.providers.get('openai'):
    assert openai.api_key
    os.environ['OPENAI_API_KEY'] = openai.api_key
