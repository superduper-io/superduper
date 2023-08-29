import inspect
import os
import time
from threading import Lock

import fil
import pytest

import superduperdb as s
from superduperdb.misc import superduper

_sleep = time.sleep

SCOPE = 'function'
LOCK = Lock()
SLEEPS = {}
SLEEP_FILE = s.ROOT / 'test/sleep.json'
MAX_SLEEP_TIME = float('inf')
AUTOUSE = True

SDDB_USE_MONGOMOCK = 'SDDB_USE_MONGOMOCK' in os.environ
SDDB_INSTRUMENT_TIME = 'SDDB_INSTRUMENT_TIME' in os.environ


@pytest.fixture(autouse=SDDB_INSTRUMENT_TIME, scope=SCOPE)
def patch_sleep(monkeypatch):
    monkeypatch.setattr(time, 'sleep', sleep)


def sleep(t: float) -> None:
    _write(t)
    _sleep(min(t, MAX_SLEEP_TIME))


def _write(t):
    stack = inspect.stack()
    if len(stack) <= 2:
        return
    module = inspect.getmodule(stack[2][0]).__file__
    with LOCK:
        entry = SLEEPS.setdefault(module, [0, 0])
        entry[0] += 1
        entry[1] += t
        fil.write(SLEEPS, SLEEP_FILE, indent=2)


@pytest.fixture(autouse=SDDB_USE_MONGOMOCK)
def patch_mongomock(monkeypatch):
    import gridfs
    import gridfs.grid_file
    import pymongo
    from mongomock import Collection, Database, MongoClient

    from superduperdb.db.base.backends import CONNECTIONS

    monkeypatch.setattr(gridfs, 'Collection', Collection)
    monkeypatch.setattr(gridfs.grid_file, 'Collection', Collection)
    monkeypatch.setattr(gridfs, 'Database', Database)
    monkeypatch.setattr(superduper, 'Database', Database)
    monkeypatch.setattr(pymongo, 'MongoClient', MongoClient)

    monkeypatch.setitem(CONNECTIONS, 'pymongo', MongoClient)
