import json
import os


try:
    with open('config.json') as f:
        cf = json.load(f)
except FileNotFoundError:  # pragma: no cover
    try:
        with open(f'{os.environ["HOME"]}/.superduperdb/config.json') as f:
            cf = json.load(f)
    except FileNotFoundError:
        cf = {'mongodb': {'host': 'localhost', 'port': 27017}, 'remote': False}

from .version import __version__
