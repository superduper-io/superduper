import random

from superduperdb.mongodb.client import SuperDuperClient
from tests.material.models import BinaryClassifier, BinaryTarget
from tests.material.types import FloatTensor, Image
from tests.material.measures import css
from tests.material.metrics import PatK, accuracy
from tests.material.losses import ranking_loss

import pytest
import torch


the_client = SuperDuperClient()

n_data_points = 250


@pytest.fixture()
def empty():
    yield the_client.test_db.documents
    the_client.drop_database('test_db', force=True)


@pytest.fixture()
def measure(empty):
    empty.create_measure('css', css)
    yield empty
    empty.delete_measure('css', force=True)


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
def my_rank_obj(empty):
    empty.create_objective('rank_obj', ranking_loss)
    yield empty
    empty.delete_objective('rank_obj', force=True)


@pytest.fixture()
def my_class_obj(empty):
    empty.create_objective('class_obj', torch.nn.BCEWithLogitsLoss())
    yield empty
    empty.delete_objective('class_obj', force=True)


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
def an_update():
    data = []
    for i in range(10):
        x = torch.randn(32)
        y = int(random.random() > 0.5)
        z = torch.randn(32)
        data.append({'x': x, 'z': z, 'y': y, 'update': True})
    return data


@pytest.fixture()
def with_semantic_index(random_data, a_model, measure):
    random_data.create_semantic_index(
        'test_semantic_index',
        ['linear_a'],
        keys=['x'],
        measure='css',
    )
    random_data.database['_meta'].insert_one({'key': 'semantic_index', 'collection': 'documents',
                                              'value': 'test_semantic_index'})
    yield random_data
    if random_data.remote:
        random_data.clear_remote_cache()
    random_data.delete_semantic_index('test_semantic_index', force=True)
    random_data['_meta'].delete_one({'key': 'semantic_index'})


@pytest.fixture()
def si_validation(random_data):
    random_data.create_validation_set('my_valid', {'_fold': 'valid'}, chunk_size=100,
                                      splitter=lambda r: ({'x': r['x'], '_fold': 'valid'}, {'z': r['z']}))

    yield random_data


@pytest.fixture()
def imputation_validation(random_data):
    random_data.create_validation_set('my_imputation_valid', chunk_size=100)
    yield random_data


@pytest.fixture()
def float_tensors(empty):
    empty.create_type('float_tensor', FloatTensor())
    t = empty.types['float_tensor']
    yield empty
    empty.delete_type('float_tensor', force=True)


@pytest.fixture()
def image_type(empty):
    empty.create_type('image', Image())
    yield empty
    empty.delete_type('image', force=True)


@pytest.fixture()
def a_model(float_tensors):
    float_tensors.create_model('linear_a', torch.nn.Linear(32, 16), type='float_tensor')
    yield float_tensors
    try:
        float_tensors.delete_model('linear_a', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def a_watcher(a_model):
    a_model.create_watcher('linear_a/x', 'linear_a', 'x')
    yield a_model
    a_model.delete_watcher('linear_a/x', force=True)


@pytest.fixture()
def a_classifier(float_tensors):
    float_tensors.create_model('classifier',
                               BinaryClassifier(32))
    yield float_tensors
    try:
        float_tensors.delete_model('classifier', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def a_target(float_tensors):
    float_tensors.create_function('target',
                                  BinaryTarget())
    yield float_tensors
    try:
        float_tensors.delete_function('target', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def b_model(float_tensors):
    float_tensors.create_model('linear_b', torch.nn.Linear(16, 8), type='float_tensor')
    yield float_tensors
    try:
        float_tensors.delete_model('linear_b', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def c_model(float_tensors):
    float_tensors.create_model('linear_c', torch.nn.Linear(32, 16), type='float_tensor')
    yield float_tensors
    try:
        float_tensors.delete_model('linear_c', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


