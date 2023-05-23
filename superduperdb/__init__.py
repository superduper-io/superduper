from pathlib import Path
import json
import os
from . misc import configs

__all__ = 'CFG', 'ICON', 'ROOT', 'cf',

CFG = configs.CONFIG.config
ICON = '🔮'
ROOT = Path(__file__).parent

try:
    with open('config.json') as f:
        cf = json.load(f)
except FileNotFoundError:  # pragma: no cover
    try:
        with open(f'{os.environ["HOME"]}/.superduperdb/config.json') as f:
            cf = json.load(f)
    except FileNotFoundError:
        cf = {'mongodb': {'host': 'localhost', 'port': 27017}, 'remote': False}

if 'openai' in cf.get('apis', {}):
    assert 'api_key' in cf['apis']['openai']
    os.environ['OPENAI_API_KEY'] = cf['apis']['openai']['api_key']
