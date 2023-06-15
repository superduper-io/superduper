# ruff: noqa: F401, F811
import PIL.PngImagePlugin
import pytest
import torch

from superduperdb.core.base import Placeholder
from superduperdb.core.documents import Document
from superduperdb.core.exceptions import ComponentInUseError, ComponentInUseWarning
from superduperdb.core.learning_task import LearningTask
from superduperdb.core.type import Type
from superduperdb.core.watcher import Watcher
from superduperdb.datalayer.mongodb.query import Select, Insert, Update, Delete
from superduperdb.models.torch.wrapper import SuperDuperModule
from superduperdb.training.torch.trainer import TorchTrainerConfiguration
from superduperdb.training.validation import validate_vector_search
from superduperdb.types.torch.tensor import tensor
from superduperdb.vector_search import VanillaHashSet
from superduperdb.vector_search.vanilla.measures import css

from tests.fixtures.collection import (
    with_vector_index,
    random_data,
    empty,
    float_tensors_8,
    float_tensors_16,
    float_tensors_32,
    a_model,
    b_model,
    a_watcher,
    an_update,
    n_data_points,
    image_type,
    si_validation,
    c_model,
    metric,
    with_vector_index_faiss,
)
from tests.material.losses import ranking_loss

IMAGE_URL = 'https://www.superduperdb.com/logos/white.png'


def test_create_component(empty, float_tensors_16, float_tensors_32):
    empty.add(SuperDuperModule(torch.nn.Linear(16, 32), 'my-test-module'))
    assert 'my-test-module' in empty.show('model')
    model = empty.models['my-test-module']
    output = model.predict_one(torch.randn(16))
    assert output.shape[0] == 32


def test_update_component(empty):
    empty.add(SuperDuperModule(torch.nn.Linear(16, 32), 'my-test-module'))
    m = SuperDuperModule(torch.nn.Linear(16, 32), 'my-test-module')
    empty.add(m)
    assert empty.show('model', 'my-test-module') == [0, 1]
    empty.add(m)
    assert empty.show('model', 'my-test-module') == [0, 1]

    n = empty.models[m.identifier]
    empty.add(n)
    assert empty.show('model', 'my-test-module') == [0, 1]


def test_compound_component(empty):
    t = tensor(torch.float, shape=(32,))

    m = SuperDuperModule(
        layer=torch.nn.Linear(16, 32),
        identifier='my-test-module',
        type=t,
    )

    empty.add(m)
    assert 'torch.float32[32]' in empty.show('type')
    assert 'my-test-module' in empty.show('model')
    assert empty.show('model', 'my-test-module') == [0]

    empty.add(m)
    assert empty.show('model', 'my-test-module') == [0]
    assert empty.show('type', 'torch.float32[32]') == [0]

    empty.add(
        SuperDuperModule(
            layer=torch.nn.Linear(16, 32),
            identifier='my-test-module',
            type=t,
        )
    )
    assert empty.show('model', 'my-test-module') == [0, 1]
    assert empty.show('type', 'torch.float32[32]') == [0]

    m = empty.load(variety='model', identifier='my-test-module', repopulate=False)
    assert isinstance(m.type, Placeholder)

    m = empty.load(variety='model', identifier='my-test-module', repopulate=True)
    assert isinstance(m.type, Type)

    with pytest.raises(ComponentInUseError):
        empty.remove('type', 'torch.float32[32]')

    with pytest.warns(ComponentInUseWarning):
        empty.remove('type', 'torch.float32[32]', force=True)

    # checks that can reload hidden type if part of another component
    m = empty.load(variety='model', identifier='my-test-module', repopulate=True)
    assert isinstance(m.type, Type)

    empty.remove('model', 'my-test-module', force=True)


def test_select_vanilla(random_data):
    r = next(random_data.execute(Select(collection='documents')))
    print(r)


def test_select(with_vector_index):
    db = with_vector_index
    r = next(db.execute(Select(collection='documents')))
    s = next(
        db.execute(
            Select(
                collection='documents',
                like=Document({'x': r['x']}),
                vector_index='test_vector_search',
            ),
        )
    )
    assert r['_id'] == s['_id']


def test_validate_component(with_vector_index, si_validation, metric):
    with_vector_index.validate(
        'test_vector_search',
        variety='vector_index',
        metrics=['p@1'],
        validation_sets=['my_valid'],
    )


def test_select_faiss(with_vector_index_faiss):
    db = with_vector_index_faiss
    r = next(db.execute(Select(collection='documents')))
    s = next(
        db.execute(
            Select(
                collection='documents',
                like=Document({'x': r['x']}),
                vector_index='test_vector_search',
            ),
        )
    )
    assert r['_id'] == s['_id']


def test_insert(random_data, a_watcher, an_update):
    random_data.execute(Insert(collection='documents', documents=an_update))
    r = next(random_data.execute(Select('documents', filter={'update': True})))
    assert 'linear_a' in r['_outputs']['x']
    assert random_data.documents.count_documents({}) == n_data_points + 10


def test_insert_from_uris(empty, image_type):
    to_insert = [
        Document(
            {
                'item': {
                    '_content': {
                        'uri': IMAGE_URL,
                        'type': 'pil_image',
                    }
                },
                'other': {
                    'item': {
                        '_content': {
                            'uri': IMAGE_URL,
                            'type': 'pil_image',
                        }
                    }
                },
            }
        )
        for _ in range(2)
    ]
    empty.execute(Insert(collection='documents', documents=to_insert))
    r = next(empty.execute(Select('documents')))
    assert isinstance(r['item'].x, PIL.PngImagePlugin.PngImageFile)
    assert isinstance(r['other']['item'].x, PIL.PngImagePlugin.PngImageFile)


def test_update(random_data, a_watcher):
    to_update = torch.randn(32)
    t = random_data.types['torch.float32[32]']
    random_data.execute(
        Update(
            collection='documents',
            filter={},
            update=Document({'$set': {'x': t(to_update)}}),
        )
    )
    cur = random_data.execute(Select('documents'))
    r = next(cur)
    s = next(cur)

    assert r['x'].x.tolist() == to_update.tolist()
    assert s['x'].x.tolist() == to_update.tolist()
    assert (
        r['_outputs']['x']['linear_a'].x.tolist()
        == s['_outputs']['x']['linear_a'].x.tolist()
    )


def test_watcher(random_data, a_model, b_model):
    random_data.add(Watcher(model='linear_a', select=Select('documents'), key='x'))
    r = next(random_data.execute(Select('documents', one=True)))
    assert 'linear_a' in r['_outputs']['x']

    t = random_data.types['torch.float32[32]']

    random_data.execute(
        Insert(
            collection='documents',
            documents=[
                Document({'x': t(torch.randn(32)), '_update': True}) for _ in range(5)
            ],
        )
    )
    r = next(random_data.execute(Select('documents', filter={'_update': True})))
    assert 'linear_a' in r['_outputs']['x']

    random_data.add(
        Watcher(
            model='linear_b',
            select=Select('documents'),
            key='x',
            features={'x': 'linear_a'},
        )
    )
    r = next(random_data.execute(Select('documents')))
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
        compute_metrics=validate_vector_search,
        hash_set_cls=VanillaHashSet,
        measure=css,
    )

    si_validation.add(configuration)
    learning_task = LearningTask(
        'my_index',
        models=['linear_a', 'linear_c'],
        select=Select('documents'),
        keys=['x', 'z'],
        metrics=['p@1'],
        training_configuration='ranking_task_parametrization',
        validation_sets=['my_valid'],
    )

    si_validation.add(learning_task)


def test_predict(a_model, float_tensors_32, float_tensors_16):
    t = float_tensors_32.types['torch.float32[32]']
    a_model.predict('linear_a', Document(t(torch.randn(32))))


def test_delete(random_data):
    r = next(random_data.execute(Select('documents')))
    random_data.execute(Delete(collection='documents', filter={'_id': r['_id']}))
    with pytest.raises(StopIteration):
        next(random_data.execute(Select('documents', filter={'_id': r['_id']})))


def test_replace(random_data):
    r = next(random_data.execute(Select('documents')))
    x = torch.randn(32)
    t = random_data.types['torch.float32[32]']
    r['x'] = t(x)
    random_data.execute(
        Update(
            collection='documents',
            filter={'_id': r['_id']},
            replacement=r,
        )
    )
    r = next(random_data.execute(Select('documents')))
    assert r['x'].x.tolist() == x.tolist()
