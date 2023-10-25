import inspect
import os
import time
from threading import Lock
from typing import Iterator

import fil
import pytest
from tenacity import Retrying, stop_after_delay

import superduperdb as s
from superduperdb.base.datalayer import Datalayer
from superduperdb.misc import superduper

_sleep = time.sleep

SCOPE = 'function'
LOCK = Lock()
SLEEPS = {}
SLEEP_FILE = s.ROOT / 'test/sleep.json'
MAX_SLEEP_TIME = float('inf')

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

    from superduperdb.backends.base.backends import CONNECTIONS

    monkeypatch.setattr(gridfs, 'Collection', Collection)
    monkeypatch.setattr(gridfs.grid_file, 'Collection', Collection)
    monkeypatch.setattr(gridfs, 'Database', Database)
    monkeypatch.setattr(superduper, 'Database', Database)
    monkeypatch.setattr(pymongo, 'MongoClient', MongoClient)

    monkeypatch.setitem(CONNECTIONS, 'pymongo', MongoClient)


@pytest.fixture
def test_db() -> Iterator[Datalayer]:
    from superduperdb import CFG
    from superduperdb.base.build import build_datalayer

    CFG.data_backend = (
        'mongodb://testmongodbuser:testmongodbpassword@localhost:27018/test_db'
    )
    for attempt in Retrying(stop=stop_after_delay(15)):
        with attempt:
            db = build_datalayer(CFG)
            db.databackend.conn.is_mongos
            print("Connected to DB instance with MongoDB!")
    yield db
    for database_name in db.databackend.conn.list_database_names():
        if database_name in ("admin", "config", "local"):
            continue
        db.databackend.conn.drop_database(database_name)
