import importlib

from superduperdb import cf

api_cf = cf.get('apis', {'providers': {}})


def client_init():
    for provider in api_cf.get('providers', {}):
        importlib.import_module(f'superduperdb.apis.{provider}.init').init_fn()
