import json
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from test.material.metrics import PatK
from test.material.models import BinaryClassifier
from typing import Iterator
from unittest import mock

import lorem
import numpy
import pymongo
import pytest
import torch
from pymongo import MongoClient
from tenacity import RetryError, Retrying, stop_after_delay

from superduperdb.base.config import DataLayer, DataLayers
from superduperdb.base.config import MongoDB as MongoDBConfig
from superduperdb.container.dataset import Dataset
from superduperdb.container.document import Document
from superduperdb.container.listener import Listener
from superduperdb.container.metric import Metric
from superduperdb.container.vector_index import VectorIndex
from superduperdb.db.base.db import DB
from superduperdb.db.mongodb.query import Collection
from superduperdb.ext.numpy.array import array
from superduperdb.ext.pillow.image import pil_image
from superduperdb.ext.torch.model import TorchModel
from superduperdb.ext.torch.tensor import tensor

n_data_points = 250


IBIS_QUERIES = {
    'insert': lambda x: x,
    'delete': ...,
}


@dataclass(frozen=True)
class TestMongoDBConfig:
    host: str = "localhost"
    port: int = 27018
    username: str = field(repr=False, default="testmongodbuser")
    password: str = field(repr=False, default="testmongodbpassword")
    serverSelectionTimeoutMS: float = 5.0


@contextmanager
def create_mongodb_client(config: TestMongoDBConfig) -> Iterator[pymongo.MongoClient]:
    client: pymongo.MongoClient
    with pymongo.MongoClient(
        host=config.host,
        port=config.port,
        username=config.username,
        password=config.password,
        serverSelectionTimeoutMS=int(config.serverSelectionTimeoutMS * 1000),
    ) as client:
        yield client


def wait_for_mongodb(config: TestMongoDBConfig, *, timeout_s: float = 30) -> None:
    try:
        for attempt in Retrying(stop=stop_after_delay(timeout_s)):
            with attempt:
                with create_mongodb_client(config) as client:
                    client.is_mongos
                    return
            print("Waiting for mongodb to start...")
    except RetryError:
        pytest.fail("Could not connect to mongodb")


def cleanup_mongodb(config: TestMongoDBConfig) -> None:
    with create_mongodb_client(config) as client:
        for database_name in client.list_database_names():
            if database_name in ("admin", "config", "local"):
                continue
            client.drop_database(database_name)


@pytest.fixture(scope='package')
def mongodb_config() -> TestMongoDBConfig:
    return TestMongoDBConfig()


@pytest.fixture(scope='package')
def _mongodb_server(mongodb_config: TestMongoDBConfig) -> Iterator[TestMongoDBConfig]:
    wait_for_mongodb(mongodb_config)
    yield mongodb_config


@pytest.fixture
def mongodb_server(_mongodb_server: TestMongoDBConfig) -> Iterator[TestMongoDBConfig]:
    # we are cleaning up the database before each test because in case of a test failure
    # one might want to inspect the state of the database
    cleanup_mongodb(_mongodb_server)
    yield _mongodb_server


@pytest.fixture
def mongodb_client(mongodb_server: TestMongoDBConfig) -> Iterator[pymongo.MongoClient]:
    with create_mongodb_client(mongodb_server) as client:
        yield client


@contextmanager
def create_datalayer(*, mongodb_config: MongoDBConfig) -> Iterator[DB]:
    from superduperdb.db.base.build import build_datalayer

    mongo_client = MongoClient(
        host=mongodb_config.host,
        port=mongodb_config.port,
        username=mongodb_config.username,
        password=mongodb_config.password,
        serverSelectionTimeoutMS=int(mongodb_config.serverSelectionTimeoutMS * 1000),
    )
    with mongo_client:
        yield build_datalayer(
            pymongo=mongo_client,
        )


@pytest.fixture
def test_db(mongodb_server: MongoDBConfig) -> Iterator[DB]:
    with create_datalayer(mongodb_config=mongodb_server) as db:
        yield db


@pytest.fixture(autouse=True)
def config(mongodb_server: MongoDBConfig) -> Iterator[None]:
    kwargs = asdict(TestMongoDBConfig())
    data_layers_cfg = DataLayers(
        artifact=DataLayer(name='_filesystem:test_db', kwargs=kwargs),
        data_backend=DataLayer(name='test_db', kwargs=kwargs),
        metadata=DataLayer(name='test_db', kwargs=kwargs),
    )

    with mock.patch('superduperdb.CFG.data_layers', data_layers_cfg):
        yield


@pytest.fixture()
def empty(test_db: DB):
    yield test_db


@pytest.fixture()
def metric(empty):
    empty.add(Metric(identifier='p@1', object=PatK(1)))
    yield
    empty.remove('metric', 'p@1', force=True)


@pytest.fixture
def random_data_factory(float_tensors_32):
    float_tensor = float_tensors_32.encoders['torch.float32[32]']

    def _factory(number_data_points=n_data_points):
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
        float_tensors_32.execute(
            Collection(name='documents').insert_many(data, refresh=False)
        )
        return float_tensors_32

    return _factory


@pytest.fixture()
def random_data(random_data_factory):
    return random_data_factory()


@pytest.fixture()
def random_arrays(arrays):
    float_array = arrays.encoders['numpy.float32[32]']
    data = []
    for i in range(n_data_points):
        x = numpy.random.randn(32).astype(numpy.float32)
        y = int(random.random() > 0.5)
        data.append(Document({'x': float_array(x), 'y': y}))

    arrays.execute(Collection(name='documents').insert_many(data, refresh=False))
    yield arrays
    arrays.execute(Collection(name='documents').delete_many({}))


@pytest.fixture()
def a_single_insert(float_tensors_32):
    float_tensor = float_tensors_32.encoders['torch.float32[32]']
    x = torch.randn(32)
    y = int(random.random() > 0.5)
    z = torch.randn(32)
    return Document(
        {
            'x': float_tensor(x),
            'y': y,
            'z': float_tensor(z),
        }
    )


@pytest.fixture()
def an_insert(float_tensors_32):
    float_tensor = float_tensors_32.encoders['torch.float32[32]']
    data = []
    for i in range(10):
        x = torch.randn(32)
        y = int(random.random() > 0.5)
        z = torch.randn(32)
        data.append(
            Document(
                {
                    'x': float_tensor(x),
                    'y': y,
                    'z': float_tensor(z),
                    'update': False,
                }
            )
        )
    return data


@pytest.fixture()
def an_update(float_tensors_32):
    float_tensor = float_tensors_32.encoders['torch.float32[32]']
    data = []
    for i in range(10):
        x = torch.randn(32)
        y = int(random.random() > 0.5)
        z = torch.randn(32)
        data.append(
            Document(
                {
                    'x': float_tensor(x),
                    'y': y,
                    'z': float_tensor(z),
                    'update': True,
                }
            )
        )

    return data


@pytest.fixture
def vector_index_factory(a_model):
    def _factory(db, identifier, **kwargs) -> VectorIndex:
        db.add(
            Listener(
                select=Collection(name='documents').find(),
                key='x',
                model='linear_a',
            )
        )
        db.add(
            Listener(
                select=Collection(name='documents').find(),
                key='z',
                model='linear_a',
            )
        )
        vi = VectorIndex(
            identifier=identifier,
            indexing_listener='linear_a/x',
            compatible_listener='linear_a/z',
            **kwargs,
        )
        db.add(vi)
        return vi

    return _factory


@pytest.fixture()
def with_vector_index(random_data, vector_index_factory):
    vector_index_factory(random_data, 'test_vector_search')
    yield random_data


@pytest.fixture()
def si_validation(random_data):
    d = Dataset(
        identifier='my_valid',
        select=Collection(name='documents').find({'_fold': 'valid'}),
        sample_size=100,
    )
    random_data.add(d)
    yield random_data


@pytest.fixture()
def float_tensors_32(empty):
    empty.add(tensor(torch.float, shape=(32,)))
    yield empty
    empty.remove('encoder', 'torch.float32[32]', force=True)


@pytest.fixture()
def float_tensors_16(empty):
    empty.add(tensor(torch.float, shape=(16,)))
    yield empty
    empty.remove('encoder', 'torch.float32[16]', force=True)


@pytest.fixture()
def float_tensors_8(empty):
    empty.add(tensor(torch.float, shape=(8,)))
    yield empty
    empty.remove('encoder', 'torch.float32[8]', force=True)


@pytest.fixture()
def arrays(empty):
    empty.add(array('float32', shape=(32,)))
    yield empty
    empty.remove('encoder', 'numpy.float32[32]', force=True)


@pytest.fixture()
def sentences(empty):
    data = []
    for _ in range(100):
        data.append(Document({'text': lorem.sentence()}))
    empty.execute(Collection(name='documents').insert_many(data))
    yield empty


@pytest.fixture()
def nursery_rhymes(empty):
    with open('test/material/data/rhymes.json') as f:
        data = json.load(f)
    for i in range(len(data)):
        data[i] = Document({'text': data[i]})
    empty.execute(Collection(name='documents').insert_many(data))
    yield empty


@pytest.fixture()
def image_type(empty):
    empty.add(pil_image)
    yield empty
    empty.remove('encoder', 'pil_image', force=True)


@pytest.fixture()
def a_model(float_tensors_32, float_tensors_16):
    float_tensors_32.add(
        TorchModel(
            object=torch.nn.Linear(32, 16),
            identifier='linear_a',
            encoder='torch.float32[16]',
        )
    )
    yield float_tensors_32
    try:
        float_tensors_32.remove('model', 'linear_a', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def a_model_base(float_tensors_32, float_tensors_16):
    float_tensors_32.add(
        TorchModel(
            object=torch.nn.Linear(32, 16),
            identifier='linear_a_base',
            encoder='torch.float32[16]',
            preprocess=lambda r: r['x'],
        ),
    )
    yield float_tensors_32
    try:
        float_tensors_32.remove('model', 'linear_a_base', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def a_listener(a_model):
    a_model.distributed = False
    a_model.add(
        Listener(
            model='linear_a',
            select=Collection(name='documents').find(),
            key='x',
        )
    )
    yield a_model
    a_model.remove('listener', 'linear_a/x', force=True)


@pytest.fixture()
def a_listener_base(a_model_base):
    a_model_base.add(
        Listener(
            model='linear_a_base',
            select=Collection(name='documents').find({}, {'_id': 0}),
            key='_base',
        )
    )
    yield a_model_base
    a_model_base.remove('listener', 'linear_a_base/_base', force=True)


@pytest.fixture()
def a_classifier(float_tensors_32):
    float_tensors_32.add(
        TorchModel(
            object=BinaryClassifier(32),
            identifier='classifier',
        ),
    )
    yield float_tensors_32
    try:
        float_tensors_32.remove('model', 'classifier', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def b_model(float_tensors_32, float_tensors_16, float_tensors_8):
    float_tensors_32.add(
        TorchModel(
            object=torch.nn.Linear(16, 8),
            identifier='linear_b',
            encoder='torch.float32[8]',
        ),
    )
    yield float_tensors_32
    try:
        float_tensors_32.remove('model', 'linear_b', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def c_model(float_tensors_32, float_tensors_16):
    float_tensors_32.add(
        TorchModel(
            object=torch.nn.Linear(32, 16),
            identifier='linear_c',
            encoder='torch.float32[16]',
        ),
    )
    yield float_tensors_32
    try:
        float_tensors_32.remove('model', 'linear_c', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e
