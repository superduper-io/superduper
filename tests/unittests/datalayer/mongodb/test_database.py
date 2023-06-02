import PIL.PngImagePlugin
import pytest
import torch

from superduperdb.core.learning_task import LearningTask
from superduperdb.core.watcher import Watcher
from superduperdb.datalayer.mongodb.query import Select, Insert, Update, Delete
from superduperdb.models.torch.wrapper import SuperDuperModule
from superduperdb.training.torch.trainer import TorchTrainerConfiguration
from superduperdb.training.validation import validate_semantic_index
from superduperdb.vector_search import VanillaHashSet
from superduperdb.vector_search.vanilla.measures import css

from tests.fixtures.collection import with_vector_index, random_data, float_tensors, empty, \
    a_model, b_model, a_watcher, an_update, n_data_points, image_type, si_validation, c_model, metric, \
    with_vector_index_faiss
from tests.material.losses import ranking_loss

IMAGE_URL = 'https://www.superduperdb.com/logos/white.png'


def test_create_component(empty):
    empty.database.create_component(SuperDuperModule(torch.nn.Linear(16, 32), 'my-test-module'))
    assert 'my-test-module' in empty.database.list_components('model')
    model = empty.database.models['my-test-module']
    output = model.predict_one(torch.randn(16))
    assert output.shape[0] == 32


def test_select(with_vector_index):
    db = with_vector_index.database
    r = next(db.select(Select(collection='documents')))
    s = next(
        db.select(
            Select(collection='documents'),
            like={'x': r['x']},
            vector_index='test_vector_search',
        )
    )
    assert r['_id'] == s['_id']


def test_select_faiss(with_vector_index_faiss):
    db = with_vector_index_faiss.database
    r = next(db.select(Select(collection='documents')))
    s = next(
        db.select(
            Select(collection='documents'),
            like={'x': r['x']},
            vector_index='test_vector_search',
        )
    )
    assert r['_id'] == s['_id']


def test_insert(random_data, a_watcher, an_update):
    random_data.database.insert(Insert(collection='documents', documents=an_update))
    r = next(random_data.database.select(Select('documents', filter={'update': True})))
    assert 'linear_a' in r['_outputs']['x']
    assert random_data.count_documents({}) == n_data_points + 10


def test_insert_from_uris(empty, image_type):
    to_insert = [
        {
            'item': {
                '_content': {
                    'uri': IMAGE_URL,
                    'type': 'image',
                }
            },
            'other': {
                'item': {
                    '_content': {
                        'uri': IMAGE_URL,
                        'type': 'image',
                    }
                }
            },
        }
        for _ in range(2)
    ]
    empty.database.insert(Insert(collection='documents', documents=to_insert))
    r = next(empty.database.select(Select('documents')))
    assert isinstance(r['item'], PIL.PngImagePlugin.PngImageFile)
    assert isinstance(
        r['other']['item'], PIL.PngImagePlugin.PngImageFile
    )


def test_update(random_data, a_watcher):
    to_update = torch.randn(32)
    random_data.database.update(
        Update(collection='documents', filter={}, update={'$set': {'x': to_update}})
    )
    cur = random_data.database.select(Select('documents'))
    r = next(cur)
    s = next(cur)

    assert r['x'].tolist() == to_update.tolist()
    assert s['x'].tolist() == to_update.tolist()
    assert r['_outputs']['x']['linear_a'].tolist() == s['_outputs']['x']['linear_a'].tolist()


def test_watcher(random_data, a_model, b_model):
    random_data.database.create_component(Watcher(model='linear_a', select=Select('documents'), key='x'))
    r = next(random_data.database.select(Select('documents', one=True)))
    assert 'linear_a' in r['_outputs']['x']

    random_data.database.insert(
        Insert('documents', documents=[{'x': torch.randn(32), 'update': True} for _ in range(5)])
    )
    r = next(random_data.database.select(Select('documents', filter={'update': True})))
    assert 'linear_a' in r['_outputs']['x']

    random_data.database.create_component(
        Watcher(model='linear_b', select=Select('documents'), key='x', features={'x': 'linear_a'})
    )
    r = next(random_data.database.select(Select('documents')))
    assert 'linear_b' in r['_outputs']['x']


def test_learning_task(si_validation, a_model, c_model, metric):

    configuration = TorchTrainerConfiguration(
        'ranking_task_parametrization',
        objective=ranking_loss,
        n_iterations=4,
        validation_interval=20,
        loader_kwargs={'batch_size': 10, 'num_workers': 0},
        optimizer_classes={
            'linear_a': torch.optim.Adam,
            'linear_c': torch.optim.Adam,
        },
        optimizer_kwargs={
            'linear_a': {'lr': 0.001},
            'linear_c': {'lr': 0.001},
        },
        compute_metrics=validate_semantic_index,
        hash_set_cls=VanillaHashSet,
        measure=css,
    )

    si_validation.database.create_component(configuration)
    learning_task = LearningTask(
        'my_index',
        models=['linear_a', 'linear_c'],
        select=Select('documents'),
        keys=['x', 'z'],
        metrics=['p@1'],
        training_configuration='ranking_task_parametrization',
        validation_sets=['my_valid'],
    )

    si_validation.database.create_component(learning_task)


def test_predict(a_model):
    a_model.database.predict_one('linear_a', torch.randn(32))


def test_delete(random_data):
    r = next(random_data.database.select(Select('documents')))
    random_data.database.delete(Delete(collection='documents', filter={'_id': r['_id']}))
    with pytest.raises(StopIteration):
        next(random_data.database.select(Select('documents', filter={'_id': r['_id']})))


def test_replace(random_data):
    r = next(random_data.database.select(Select('documents')))
    x = torch.randn(32)
    r['x'] = x
    random_data.database.update(
        Update(
            collection='documents',
            filter={'_id': r['_id']},
            replacement=r,
        )
    )
    r = next(random_data.database.select(Select('documents')))
    assert r['x'].tolist() == x.tolist()
