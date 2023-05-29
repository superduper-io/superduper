import random

import lorem
import numpy

from superduperdb.datalayer.mongodb.client import SuperDuperClient
from superduperdb.models.torch.wrapper import SuperDuperModule
from superduperdb.vector_search.vanilla.hashes import VanillaHashSet
from tests.material.models import BinaryClassifier, BinaryTarget, LinearBase
from tests.material.types import FloatTensor, Image, Array32, Int64
from tests.material.measures import css
from tests.material.metrics import PatK, accuracy

import pytest
import torch


n_data_points = 250


@pytest.fixture()
def empty(client: SuperDuperClient):
    db = client.test_db.documents
    db.remote = False
    yield db
    client.drop_database('test_db', force=True)


@pytest.fixture()
def metric(empty):
    empty.create_metric('p_at_1', PatK(1))
    yield empty
    empty.delete_metric('p_at_1', force=True)


@pytest.fixture()
def accuracy_metric(empty):
    empty.create_metric('accuracy_metric', accuracy)
    yield empty
    empty.delete_metric('accuracy_metric', force=True)


@pytest.fixture()
def random_data(float_tensors):
    data = []
    for i in range(n_data_points):
        x = torch.randn(32)
        y = int(random.random() > 0.5)
        z = torch.randn(32)
        data.append({'x': x, 'z': z, 'y': y})
    float_tensors.insert_many(data, refresh=False)
    yield float_tensors
    float_tensors.delete_many({})


@pytest.fixture()
def random_arrays(arrays):
    data = []
    for i in range(n_data_points):
        x = numpy.random.randn(32).astype(numpy.float32)
        y = int(random.random() > 0.5)
        data.append({'x': x, 'y': y})
    arrays.insert_many(data, refresh=False)
    yield arrays
    arrays.delete_many({})


@pytest.fixture()
def an_update():
    data = []
    for i in range(10):
        x = torch.randn(32)
        y = int(random.random() > 0.5)
        z = torch.randn(32)
        data.append({'x': x, 'z': z, 'y': y, 'update': True})
    return data


@pytest.fixture()
def with_semantic_index(random_data, a_model):
    random_data.create_learning_task(
        ['linear_a'],
        ['x'],
        keys_to_watch=['x'],
        identifier='test_learning_task',
        configuration={'hash_set_cls': VanillaHashSet, 'measure': css},
    )
    yield random_data
    random_data.delete_learning_task('test_learning_task', force=True)


@pytest.fixture()
def si_validation(random_data):
    random_data.create_validation_set('my_valid', {'_fold': 'valid'}, chunk_size=100)

    yield random_data


@pytest.fixture()
def imputation_validation(random_data):
    random_data.create_validation_set('my_imputation_valid', chunk_size=100)
    yield random_data


@pytest.fixture()
def float_tensors(empty):
    empty.create_type('float_tensor', FloatTensor())
    empty.types['float_tensor']
    yield empty
    empty.delete_type('float_tensor', force=True)


@pytest.fixture()
def arrays(empty):
    empty.create_type('array', Array32())
    empty.types['array']
    yield empty
    empty.delete_type('array', force=True)


@pytest.fixture()
def sentences(empty):
    data = []
    for _ in range(100):
        data.append({'text': lorem.sentence()})
    empty.insert_many(data)
    yield empty


@pytest.fixture()
def int64(empty):
    empty.create_type('int64', Int64())
    empty.types['int64']
    yield empty
    empty.delete_type('int64', force=True)


@pytest.fixture()
def image_type(empty):
    empty.create_type('image', Image())
    yield empty
    empty.delete_type('image', force=True)


@pytest.fixture()
def a_model(float_tensors):
    float_tensors.create_model(
        'linear_a',
        SuperDuperModule(torch.nn.Linear(32, 16), 'linear_a'),
        type='float_tensor',
    )
    yield float_tensors
    try:
        float_tensors.delete_model('linear_a', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def a_model_base(float_tensors):
    float_tensors.create_model(
        'linear_a_base',
        SuperDuperModule(LinearBase(32, 16), 'linear_a_base'),
        type='float_tensor',
    )
    yield float_tensors
    try:
        float_tensors.delete_model('linear_a_base', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def a_watcher(a_model):
    a_model.remote = False
    a_model.create_watcher('linear_a/x', 'linear_a', key='x')
    yield a_model
    a_model.delete_watcher('linear_a/x', force=True)


@pytest.fixture()
def a_watcher_base(a_model_base):
    a_model_base.create_watcher('linear_a_base/_base', 'linear_a_base', key='_base')
    yield a_model_base
    a_model_base.delete_watcher('linear_a_base/_base', force=True)


@pytest.fixture()
def a_classifier(float_tensors):
    float_tensors.create_model(
        'classifier',
        SuperDuperModule(BinaryClassifier(32), 'classifier'),
    )
    yield float_tensors
    try:
        float_tensors.delete_model('classifier', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def a_target(float_tensors):
    float_tensors.create_function('target', BinaryTarget())
    yield float_tensors
    try:
        float_tensors.delete_function('target', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def b_model(float_tensors):
    float_tensors.create_model(
        'linear_b',
        SuperDuperModule(torch.nn.Linear(16, 8), 'linear_b'),
        type='float_tensor'
    )
    yield float_tensors
    try:
        float_tensors.delete_model('linear_b', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def c_model(float_tensors):
    float_tensors.create_model(
        'linear_c',
        SuperDuperModule(torch.nn.Linear(32, 16), 'linear_c'),
        type='float_tensor',
    )
    yield float_tensors
    try:
        float_tensors.delete_model('linear_c', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e
