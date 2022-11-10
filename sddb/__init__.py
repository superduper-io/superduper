import json

try:
    with open('config.json') as f:
        cf = json.load(f)
except FileNotFoundError:
    cf = {'mongodb': {'host': 'localhost', 'port': 27017}}
