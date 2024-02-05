import inspect
import os
import random
import time
from pathlib import Path
from threading import Lock
from typing import Iterator

import pytest
from dotenv import load_dotenv

if env_file := os.environ.get('SUPERDUPERDB_PYTEST_ENV_FILE', ''):
    load_dotenv(env_file)

import superduperdb as s
from superduperdb import CFG, logging
from superduperdb.backends.ibis.field_types import dtype
from superduperdb.backends.ibis.query import Table
from superduperdb.backends.mongodb.data_backend import MongoDataBackend
from superduperdb.backends.mongodb.query import Collection

# ruff: noqa: E402
from superduperdb.base import config as _config, superduper
from superduperdb.base.build import build_datalayer
from superduperdb.base.datalayer import Datalayer
from superduperdb.base.document import Document
from superduperdb.components.dataset import Dataset
from superduperdb.components.listener import Listener
from superduperdb.components.schema import Schema
from superduperdb.components.vector_index import VectorIndex
from superduperdb.ext.pillow.encoder import pil_image

from .db_config import DBConfig

_config._CONFIG_IMMUTABLE = False


try:
    import torch

    from superduperdb.ext.torch.encoder import tensor
    from superduperdb.ext.torch.model import TorchModel
except ImportError:
    torch = None

GLOBAL_TEST_N_DATA_POINTS = 250
LOCAL_TEST_N_DATA_POINTS = 5

_sleep = time.sleep

SCOPE = 'function'
LOCK = Lock()
SLEEPS = {}
SLEEP_FILE = s.ROOT / 'test/sleep.json'
MAX_SLEEP_TIME = float('inf')

SDDB_USE_MONGOMOCK = 'SDDB_USE_MONGOMOCK' in os.environ
SDDB_INSTRUMENT_TIME = 'SDDB_INSTRUMENT_TIME' in os.environ
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


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

    # mongodb instead of localhost is required for CFG compatibility with docker-host
    db_name = CFG.data_backend.split('/')[-1]
    from superduperdb.base.config import Cluster

    monkeypatch.setattr(CFG, 'cluster', Cluster())

    db = build_datalayer(CFG)

    yield db

    logging.info("Dropping database ", {db_name})

    db.databackend.conn.drop_database(db_name)
    db.databackend.conn.drop_database(f'_filesystem:{db_name}')


@pytest.fixture
def valid_dataset(db):
    if isinstance(db.databackend, MongoDataBackend):
        select = Collection('documents').find({'_fold': 'valid'})
    else:
        table = db.load('table', 'documents')
        select = table.select('id', 'x', 'y', 'z').filter(table._fold == 'valid')
    d = Dataset(
        identifier='my_valid',
        select=select,
        sample_size=100,
    )
    return d


def add_random_data_to_sql_db(
    db: Datalayer,
    table_name: str = 'documents',
    number_data_points: int = GLOBAL_TEST_N_DATA_POINTS,
):
    float_tensor = db.datatypes['torch.float32[32]']
    data = []

    schema = Schema(
        identifier=table_name,
        fields={
            'id': dtype('str'),
            'x': float_tensor,
            'y': dtype('int32'),
            'z': float_tensor,
        },
    )
    t = Table(identifier=table_name, schema=schema)
    db.add(t)

    for i in range(number_data_points):
        x = torch.randn(32)
        y = int(random.random() > 0.5)
        z = torch.randn(32)
        data.append(
            Document(
                {
                    'id': str(i),
                    'x': x,
                    'y': y,
                    'z': z,
                }
            )
        )
    db.execute(
        t.insert(data),
        refresh=False,
    )


def add_random_data_to_mongo_db(
    db: Datalayer,
    collection_name: str = 'documents',
    number_data_points: int = GLOBAL_TEST_N_DATA_POINTS,
):
    float_tensor = db.datatypes['torch.float32[32]']
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

    db.execute(
        Collection(collection_name).insert_many(data),
        refresh=False,
    )


def add_datatypes(db: Datalayer):
    for n in [8, 16, 32]:
        db.add(tensor(torch.float, shape=(n,)))
    db.add(pil_image)


def add_models(db: Datalayer):
    # identifier, weight_shape, encoder
    params = [
        ['linear_a', (32, 16), db.load('datatype', 'torch.float32[16]')],
        ['linear_b', (16, 8), db.load('datatype', 'torch.float32[8]')],
    ]
    for identifier, weight_shape, datatype in params:
        m = TorchModel(
            object=torch.nn.Linear(*weight_shape),
            identifier=identifier,
            datatype=datatype,
            preferred_devices=['cpu'],
        )
        db.add(m)


def add_vector_index(
    db: Datalayer, collection_name='documents', identifier='test_vector_search'
):
    # TODO: Support configurable key and mode
    is_mongodb_backend = isinstance(db.databackend, MongoDataBackend)
    if is_mongodb_backend:
        select_x = Collection(collection_name).find()
        select_z = Collection(collection_name).find()
    else:
        table = db.load('table', collection_name).to_query()
        select_x = table.select('id', 'x')
        select_z = table.select('id', 'z')

    model = db.load('model', 'linear_a')

    _, i_list = db.add(
        Listener(
            select=select_x,
            key='x',
            model=model,
        )
    )

    _, c_list = db.add(
        Listener(
            select=select_z,
            key='z',
            model=model,
        )
    )
    vi = VectorIndex(
        identifier=identifier,
        indexing_listener=i_list,
        compatible_listener=c_list,
    )
    db.add(vi)


@pytest.fixture(scope='session')
def image_url():
    path = Path(__file__).parent / 'material' / 'data' / '1x1.png'
    return f'file://{path}'


def create_db(CFG, **kwargs):
    # TODO: support more parameters to control the setup
    db = build_datalayer(CFG)
    if kwargs.get('empty', False):
        return db

    add_datatypes(db)

    # prepare data
    n_data = kwargs.get('n_data', LOCAL_TEST_N_DATA_POINTS)
    is_mongodb_backend = isinstance(db.databackend, MongoDataBackend)
    if is_mongodb_backend:
        add_random_data_to_mongo_db(db, number_data_points=n_data)
    else:
        add_random_data_to_sql_db(db, number_data_points=n_data)

    # prepare models
    if kwargs.get('add_models', True):
        add_models(db)

    # prepare vector index
    if kwargs.get('add_vector_index', True):
        add_vector_index(db)

    return db


# Caution: This is only meant to be used in unit test cases
@pytest.fixture
def db(request, monkeypatch) -> Iterator[Datalayer]:
    # TODO: Use pre-defined config instead of dict here
    param = request.param if hasattr(request, 'param') else DBConfig.mongodb
    if isinstance(param, dict):
        # e.g. @pytest.mark.parametrize("db", [DBConfig.sqldb], indirect=True)
        setup_config = param.copy()
    elif isinstance(param, tuple):
        # e.g. @pytest.mark.parametrize(
        #          "db", [(DBConfig.sqldb, {'n_data': 10})], indirect=True)
        setup_config = param[0].copy()
        setup_config.update(param[1] or {})
    else:
        raise ValueError(f'Unsupported param type: {type(param)}')

    # This patch is required for testing sql databases
    monkeypatch.setattr(CFG, 'data_backend', setup_config['data_backend'])

    from superduperdb.base.config import Cluster

    monkeypatch.setattr(CFG, 'cluster', Cluster())
    db = create_db(CFG, **setup_config)

    yield db

    db_type = setup_config.get('db_type', 'mongodb')
    if db_type == "mongodb":
        db.drop(force=True)
    elif db_type == "sqldb":
        db.artifact_store.drop(force=True)
        tables = db.databackend.conn.list_tables()
        for table in tables:
            db.databackend.conn.drop_table(table, force=True)
