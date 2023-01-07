import json

try:
    with open('config.json') as f:
        cf = json.load(f)
except FileNotFoundError:  # pragma: no cover
    cf = {'mongodb': {'host': 'localhost', 'port': 27017}}
