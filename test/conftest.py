import inspect
import os
import random
import time
from pathlib import Path
from threading import Lock
from typing import Iterator

import pytest

import superduperdb as s
from superduperdb import logging
from superduperdb.base import superduper

try:
    import torch

    from superduperdb.ext.torch.encoder import tensor
    from superduperdb.ext.torch.model import TorchModel
except ImportError:
    torch = None

from superduperdb import CFG
from superduperdb.backends.mongodb.query import Collection
from superduperdb.base.build import build_datalayer
from superduperdb.base.datalayer import Datalayer
from superduperdb.base.document import Document
from superduperdb.components.dataset import Dataset
from superduperdb.components.listener import Listener
from superduperdb.components.vector_index import VectorIndex
from superduperdb.ext.pillow.encoder import pil_image

GLOBAL_TEST_N_DATA_POINTS = 250
LOCAL_TEST_N_DATA_POINTS = 5

MONGOMOCK_URI = 'mongomock:///test_db'


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
        with open(SLEEP_FILE, 'w') as f:
            f.write(SLEEPS, f)


@pytest.fixture(autouse=SDDB_USE_MONGOMOCK)
def patch_mongomock(monkeypatch):
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
def test_db(monkeypatch, request) -> Iterator[Datalayer]:
    from superduperdb import CFG
    from superduperdb.base.build import build_datalayer

    db_name = "test_db"
    data_backend = f'mongodb://superduper:superduper@localhost:27017/{db_name}'

    monkeypatch.setattr(CFG, 'data_backend', data_backend)

    db = build_datalayer(CFG)

    yield db

    logging.info("Dropping database ", {db_name})

    db.databackend.conn.drop_database(db_name)
    db.databackend.conn.drop_database(f'_filesystem:{db_name}')


@pytest.fixture
def valid_dataset():
    d = Dataset(
        identifier='my_valid',
        select=Collection('documents').find({'_fold': 'valid'}),
        sample_size=100,
    )
    return d


def add_random_data(
    db: Datalayer,
    collection_name: str = 'documents',
    number_data_points: int = GLOBAL_TEST_N_DATA_POINTS,
):
    float_tensor = db.encoders['torch.float32[32]']
    data = []
    for i in range(number_data_points):
        x = torch.randn(32)
        y = int(random.random() > 0.5)
        z = torch.randn(32)
        data.append(
            Document(
                {
                    'x': float_tensor(x),
                    'y': y,
                    'z': float_tensor(z),
                }
            )
        )

    if data:
        db.execute(
            Collection(collection_name).insert_many(data),
            refresh=False,
        )


def add_encoders(db: Datalayer):
    for n in [8, 16, 32]:
        db.add(tensor(torch.float, shape=(n,)))
    db.add(pil_image)


def add_models(db: Datalayer):
    # identifier, weight_shape, encoder
    params = [
        ['linear_a', (32, 16), 'torch.float32[16]'],
        ['linear_b', (16, 8), 'torch.float32[8]'],
    ]
    for identifier, weight_shape, encoder in params:
        db.add(
            TorchModel(
                object=torch.nn.Linear(*weight_shape),
                identifier=identifier,
                encoder=encoder,
            )
        )


def add_vector_index(
    db: Datalayer, collection_name='documents', identifier='test_vector_search'
):
    # TODO: Support configurable key and model
    db.add(
        Listener(
            select=Collection(collection_name).find(),
            key='x',
            model='linear_a',
        )
    )
    db.add(
        Listener(
            select=Collection(collection_name).find(),
            key='z',
            model='linear_a',
        )
    )
    vi = VectorIndex(
        identifier=identifier,
        indexing_listener='linear_a/x',
        compatible_listener='linear_a/z',
    )
    db.add(vi)


@pytest.fixture(scope='session')
def image_url():
    path = Path(__file__).parent / 'material' / 'data' / '1x1.png'
    return f'file://{path}'


def setup_db(db, **kwargs):
    # TODO: support more parameters to control the setup
    add_encoders(db)
    n_data = kwargs.get('n_data', GLOBAL_TEST_N_DATA_POINTS)
    add_random_data(db, number_data_points=n_data)
    if kwargs.get('add_models', True):
        add_models(db)
    if kwargs.get('add_vector_index', True):
        add_vector_index(db)


@pytest.fixture(scope='session')
def db() -> Datalayer:
    db = build_datalayer(CFG, data_backend=MONGOMOCK_URI)
    setup_db(db)
    return db


@pytest.fixture
def local_db(request) -> Datalayer:
    db = build_datalayer(CFG, data_backend=MONGOMOCK_URI)
    setup_config = getattr(request, 'param', {'n_data': LOCAL_TEST_N_DATA_POINTS})
    setup_db(db, **setup_config)
    return db


@pytest.fixture
def local_empty_db(request) -> Datalayer:
    db = build_datalayer(CFG, data_backend=MONGOMOCK_URI)
    return db
