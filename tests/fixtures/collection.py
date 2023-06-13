import json
import random

import lorem
import numpy
import pytest
import torch

from superduperdb.core.documents import Document
from superduperdb.core.vector_index import VectorIndex
from superduperdb.core.watcher import Watcher
from superduperdb.datalayer.mongodb.client import SuperDuperClient
from superduperdb.datalayer.mongodb.query import Select, Insert, Delete
from superduperdb.models.torch.wrapper import SuperDuperModule
from superduperdb.types.numpy.array import array
from superduperdb.types.pillow.image import pil_image
from superduperdb.types.torch.tensor import tensor
from superduperdb.vector_search import FaissHashSet
from tests.material.models import BinaryClassifier, LinearBase
from tests.material.metrics import PatK


n_data_points = 250


@pytest.fixture()
def empty(client: SuperDuperClient):
    db = client.test_db.documents
    db.remote = False
    yield db
    client.drop_database('test_db', force=True)


@pytest.fixture()
def metric(empty):
    empty.database.create_component(PatK(1))
    yield empty
    empty.database.delete_component('metric', 'p@1', force=True)


@pytest.fixture()
def random_data(float_tensors):
    float_tensor = float_tensors.database.types['torch.float32']

    data = []
    for i in range(n_data_points):
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
    float_tensors.database.insert(
        Insert(collection='documents', documents=data), refresh=False
    )
    yield float_tensors
    float_tensors.delete_many({})


@pytest.fixture()
def random_arrays(arrays):
    float_array = arrays.database.types['numpy.float32']
    data = []
    for i in range(n_data_points):
        x = numpy.random.randn(32).astype(numpy.float32)
        y = int(random.random() > 0.5)
        data.append(Document({'x': float_array(x), 'y': y}))
    arrays.database.insert(
        Insert(collection='documents', documents=data), refresh=False
    )
    yield arrays
    arrays.database.delete(Delete('documents', {}))


@pytest.fixture()
def an_update(float_tensors):
    float_tensor = float_tensors.database.types['torch.float32']
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


@pytest.fixture()
def with_vector_index(random_data, a_model):
    random_data.database.create_component(
        Watcher(select=Select('documents'), key='x', model='linear_a')
    )
    vi = VectorIndex(
        'test_vector_search',
        models=['linear_a'],
        keys=['x'],
        watcher='linear_a/x',
    )
    random_data.database.create_component(vi)
    yield random_data
    random_data.database.delete_component(
        'vector_index', 'test_vector_search', force=True
    )


@pytest.fixture()
def with_vector_index_faiss(random_data, a_model):
    random_data.database.create_component(
        Watcher(select=Select('documents'), key='x', model='linear_a')
    )
    random_data.database.create_component(
        VectorIndex(
            'test_vector_search',
            models=['linear_a'],
            keys=['x'],
            watcher='linear_a/x',
            hash_set_cls=FaissHashSet,
        )
    )
    yield random_data
    random_data.database.delete_component(
        'vector_index', 'test_vector_search', force=True
    )


@pytest.fixture()
def si_validation(random_data):
    random_data.database.create_validation_set(
        'my_valid',
        select=Select('documents', filter={'_fold': 'valid'}),
        chunk_size=100,
    )

    yield random_data


@pytest.fixture()
def imputation_validation(random_data):
    random_data.create_validation_set('my_imputation_valid', chunk_size=100)
    yield random_data


@pytest.fixture()
def float_tensors(empty):
    empty.database.create_component(tensor(torch.float))
    yield empty
    empty.database.delete_component('type', 'torch.float32', force=True)


@pytest.fixture()
def arrays(empty):
    empty.database.create_component(array('float32'))
    yield empty
    empty.database.delete_component('type', 'numpy.float32', force=True)


@pytest.fixture()
def sentences(empty):
    data = []
    for _ in range(100):
        data.append(Document({'text': lorem.sentence()}))
    empty.database.insert(Insert(collection='documents', documents=data))
    yield empty


@pytest.fixture()
def nursery_rhymes(empty):
    with open('tests/material/data/rhymes.json') as f:
        data = json.load(f)
    for i in range(len(data)):
        data[i] = Document({'text': data[i]})
    empty.database.insert(Insert(collection='documents', documents=data))
    yield empty


@pytest.fixture()
def int64(empty):
    empty.database.create_component(array(numpy.int64))
    yield empty
    empty.database.delete_component('type', 'int64', force=True)


@pytest.fixture()
def image_type(empty):
    empty.database.create_component(pil_image)
    yield empty
    empty.database.delete_component('type', 'pil_image', force=True)


@pytest.fixture()
def a_model(float_tensors):
    float_tensors.database.create_component(
        SuperDuperModule(torch.nn.Linear(32, 16), 'linear_a', type='torch.float32')
    )
    yield float_tensors
    try:
        float_tensors.database.delete_component('model', 'linear_a', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def a_model_base(float_tensors):
    float_tensors.database.create_component(
        SuperDuperModule(LinearBase(32, 16), 'linear_a_base', type='torch.float32'),
    )
    yield float_tensors
    try:
        float_tensors.database.delete_component('model', 'linear_a_base', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def a_watcher(a_model):
    a_model.remote = False
    a_model.database.create_component(
        Watcher(model='linear_a', select=Select('documents'), key='x')
    )
    yield a_model
    a_model.database.delete_component('watcher', 'linear_a/x', force=True)


@pytest.fixture()
def a_watcher_base(a_model_base):
    a_model_base.database.create_component(
        Watcher(model='linear_a_base', select=Select('documents'), key='_base')
    )
    yield a_model_base
    a_model_base.database.delete_component('watcher', 'linear_a_base/_base', force=True)


@pytest.fixture()
def a_classifier(float_tensors):
    float_tensors.database.create_component(
        SuperDuperModule(BinaryClassifier(32), 'classifier'),
    )
    yield float_tensors
    try:
        float_tensors.database.delete_component('model', 'classifier', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def b_model(float_tensors):
    float_tensors.database.create_component(
        SuperDuperModule(torch.nn.Linear(16, 8), 'linear_b', type='torch.float32'),
    )
    yield float_tensors
    try:
        float_tensors.database.delete_component('model', 'linear_b', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def c_model(float_tensors):
    float_tensors.database.create_component(
        SuperDuperModule(torch.nn.Linear(32, 16), 'linear_c', type='torch.float32'),
    )
    yield float_tensors
    try:
        float_tensors.database.delete_component('model', 'linear_c', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e
