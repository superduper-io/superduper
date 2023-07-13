import json
import multiprocessing
import random
import time
from contextlib import contextmanager
from dataclasses import asdict
from typing import Iterator
from unittest import mock

import lorem
import numpy
import pytest
import torch
from pymongo import MongoClient
from superduperdb.core.dataset import Dataset
from superduperdb.core.documents import Document
from superduperdb.core.metric import Metric
from superduperdb.core.vector_index import VectorIndex
from superduperdb.core.watcher import Watcher
from superduperdb.datalayer.base.database import BaseDatabase
from superduperdb.datalayer.mongodb.query import Collection
from superduperdb.encoders.numpy.array import array
from superduperdb.encoders.pillow.image import pil_image
from superduperdb.encoders.torch.tensor import tensor
from superduperdb.misc.config import DataLayer, DataLayers
from superduperdb.misc.config import MongoDB as MongoDBConfig
from superduperdb.models.torch.wrapper import TorchModel
from superduperdb.serve.server import serve

from tests.material.metrics import PatK
from tests.material.models import BinaryClassifier

from .conftest_mongodb import MongoDBConfig as TestMongoDBConfig

n_data_points = 250

pytest_plugins = [
    "tests.conftest_mongodb",
]


@pytest.fixture(scope="session", autouse=True)
def start_stop_server():
    p = multiprocessing.Process(target=serve)  # couldn't get this to work with threads
    p.start()
    time.sleep(3)  # need to wait for the process to initialize
    yield
    p.terminate()
    p.join()


@contextmanager
def create_datalayer(*, mongodb_config: MongoDBConfig) -> Iterator[BaseDatabase]:
    from superduperdb.datalayer.base.build import build_datalayer

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
def test_db(mongodb_server: MongoDBConfig) -> Iterator[BaseDatabase]:
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
def empty(test_db: BaseDatabase):
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
            Watcher(
                select=Collection(name='documents').find(),
                key='x',
                model='linear_a',
                db=a_model,
            )
        )
        db.add(
            Watcher(
                select=Collection(name='documents').find(),
                key='z',
                model='linear_a',
                db=a_model,
            )
        )
        vi = VectorIndex(
            identifier=identifier,
            indexing_watcher='linear_a/x',
            compatible_watchers=['linear_a/z'],
            db=a_model,
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
        db=random_data,
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
    with open('tests/material/data/rhymes.json') as f:
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
            db=float_tensors_16,
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
            db=float_tensors_16,
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
def a_watcher(a_model):
    a_model.distributed = False
    a_model.add(
        Watcher(
            model='linear_a',
            select=Collection(name='documents').find(),
            key='x',
            db=a_model,
        )
    )
    yield a_model
    a_model.remove('watcher', 'linear_a/x', force=True)


@pytest.fixture()
def a_watcher_base(a_model_base):
    a_model_base.add(
        Watcher(
            model='linear_a_base',
            select=Collection(name='documents').find({}, {'_id': 0}),
            key='_base',
            db=a_model_base,
        )
    )
    yield a_model_base
    a_model_base.remove('watcher', 'linear_a_base/_base', force=True)


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
            db=float_tensors_32,
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
            db=float_tensors_16,
        ),
    )
    yield float_tensors_32
    try:
        float_tensors_32.remove('model', 'linear_c', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e
