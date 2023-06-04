import importlib
from superduperdb import CFG


def client_init():
    for provider in CFG.apis.providers:
        importlib.import_module(f'superduperdb.apis.{provider}.init').init_fn()
