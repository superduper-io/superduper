import PIL.PngImagePlugin
import pytest
import torch

from superduperdb.container.dataset import Dataset
from superduperdb.container.document import Document
from superduperdb.container.encoder import Encoder
from superduperdb.container.listener import Listener
from superduperdb.db.base.exceptions import ComponentInUseError, ComponentInUseWarning
from superduperdb.db.mongodb.query import Collection
from superduperdb.ext.torch.model import TorchModel
from superduperdb.ext.torch.tensor import tensor

n_data_points = 250

IMAGE_URL = 'https://www.superduperdb.com/logos/white.png'


def test_create_component(
    empty_database, database_with_float_tensors_16, database_with_float_tensors_32
):
    empty_database.add(
        TorchModel(object=torch.nn.Linear(16, 32), identifier='my-test-module')
    )
    assert 'my-test-module' in empty_database.show('model')
    model = empty_database.models['my-test-module']
    print(model)
    output = model.predict(torch.randn(16), one=True)
    assert output.shape[0] == 32


def test_update_component(empty_database):
    empty_database.add(
        TorchModel(object=torch.nn.Linear(16, 32), identifier='my-test-module')
    )
    m = TorchModel(object=torch.nn.Linear(16, 32), identifier='my-test-module')
    empty_database.add(m)
    assert empty_database.show('model', 'my-test-module') == [0, 1]
    empty_database.add(m)
    assert empty_database.show('model', 'my-test-module') == [0, 1]

    n = empty_database.models[m.identifier]
    empty_database.add(n)
    assert empty_database.show('model', 'my-test-module') == [0, 1]


def test_compound_component(empty_database):
    t = tensor(torch.float, shape=(32,))

    m = TorchModel(
        object=torch.nn.Linear(16, 32),
        identifier='my-test-module',
        encoder=t,
    )

    empty_database.add(m)
    assert 'torch.float32[32]' in empty_database.show('encoder')
    assert 'my-test-module' in empty_database.show('model')
    assert empty_database.show('model', 'my-test-module') == [0]

    empty_database.add(m)
    assert empty_database.show('model', 'my-test-module') == [0]
    assert empty_database.show('encoder', 'torch.float32[32]') == [0]

    empty_database.add(
        TorchModel(
            object=torch.nn.Linear(16, 32),
            identifier='my-test-module',
            encoder=t,
        )
    )
    assert empty_database.show('model', 'my-test-module') == [0, 1]
    assert empty_database.show('encoder', 'torch.float32[32]') == [0]

    m = empty_database.load(type_id='model', identifier='my-test-module')
    assert isinstance(m.encoder, Encoder)

    with pytest.raises(ComponentInUseError):
        empty_database.remove('encoder', 'torch.float32[32]')

    with pytest.warns(ComponentInUseWarning):
        empty_database.remove('encoder', 'torch.float32[32]', force=True)

    empty_database.remove('model', 'my-test-module', force=True)


def test_select_vanilla(database_with_random_tensor_data):
    r = database_with_random_tensor_data.execute(
        Collection(name='documents').find_one()
    )
    print(r)


def test_select(database_with_vector_index):
    db = database_with_vector_index
    r = db.execute(Collection(name='documents').find_one())
    query = Collection(name='documents').like(
        r=Document({'x': r['x']}),
        vector_index='test_vector_search',
    )
    s = next(db.execute(query))
    assert r['_id'] == s['_id']


def test_reload_dataset(database_with_dataset):
    database_with_dataset.load('dataset', 'my_valid')


def test_insert(
    database_with_random_tensor_data,
    database_with_listener_torch_model_a,
    multiple_documents,
):
    database_with_random_tensor_data.execute(
        Collection(name='documents').insert_many(multiple_documents)
    )
    r = next(
        database_with_random_tensor_data.execute(
            Collection(name='documents').find({'update': True})
        )
    )
    assert 'linear_a' in r['_outputs']['x']
    assert (
        len(
            list(
                database_with_random_tensor_data.execute(
                    Collection(name='documents').find()
                )
            )
        )
        == n_data_points + 10
    )


def test_insert_from_uris(empty_database, database_with_pil_image):
    to_insert = [
        Document(
            {
                'item': {
                    '_content': {
                        'uri': IMAGE_URL,
                        'encoder': 'pil_image',
                    }
                },
                'other': {
                    'item': {
                        '_content': {
                            'uri': IMAGE_URL,
                            'encoder': 'pil_image',
                        }
                    }
                },
            }
        )
        for _ in range(2)
    ]
    empty_database.execute(Collection(name='documents').insert_many(to_insert))
    r = empty_database.execute(Collection(name='documents').find_one())
    assert isinstance(r['item'].x, PIL.PngImagePlugin.PngImageFile)
    assert isinstance(r['other']['item'].x, PIL.PngImagePlugin.PngImageFile)


def test_update(database_with_random_tensor_data, database_with_listener_torch_model_a):
    to_update = torch.randn(32)
    t = database_with_random_tensor_data.encoders['torch.float32[32]']
    database_with_random_tensor_data.execute(
        Collection(name='documents').update_many(
            {}, Document({'$set': {'x': t(to_update)}})
        )
    )
    cur = database_with_random_tensor_data.execute(Collection(name='documents').find())
    r = next(cur)
    s = next(cur)

    assert all(r['x'].x == to_update)
    assert all(s['x'].x == to_update)
    assert (
        r['_outputs']['x']['linear_a'].x.tolist()
        == s['_outputs']['x']['linear_a'].x.tolist()
    )


def test_listener(
    database_with_random_tensor_data,
    database_with_torch_model_a,
    database_with_torch_model_b,
):
    database_with_random_tensor_data.add(
        Listener(
            model='linear_a',
            select=Collection(name='documents').find(),
            key='x',
        ),
    )
    r = database_with_random_tensor_data.execute(
        Collection(name='documents').find_one()
    )
    assert 'linear_a' in r['_outputs']['x']

    t = database_with_random_tensor_data.encoders['torch.float32[32]']

    database_with_random_tensor_data.execute(
        Collection(name='documents').insert_many(
            [Document({'x': t(torch.randn(32)), 'update': True}) for _ in range(5)]
        )
    )

    r = database_with_random_tensor_data.execute(
        Collection(name='documents').find_one({'update': True})
    )
    assert 'linear_a' in r['_outputs']['x']

    database_with_random_tensor_data.add(
        Listener(
            model='linear_b',
            select=Collection(name='documents').find().featurize({'x': 'linear_a'}),
            key='x',
        )
    )
    r = database_with_random_tensor_data.execute(
        Collection(name='documents').find_one()
    )
    assert 'linear_b' in r['_outputs']['x']


def test_predict(
    database_with_torch_model_a,
    database_with_float_tensors_32,
    database_with_float_tensors_16,
):
    t = database_with_float_tensors_32.encoders['torch.float32[32]']
    database_with_torch_model_a.predict('linear_a', Document(t(torch.randn(32))))


def test_delete(database_with_random_tensor_data):
    r = database_with_random_tensor_data.execute(
        Collection(name='documents').find_one()
    )
    database_with_random_tensor_data.execute(
        Collection(name='documents').delete_many({'_id': r['_id']})
    )
    with pytest.raises(StopIteration):
        next(
            database_with_random_tensor_data.execute(
                Collection(name='documents').find({'_id': r['_id']})
            )
        )


def test_replace(database_with_random_tensor_data):
    r = next(
        database_with_random_tensor_data.execute(Collection(name='documents').find())
    )
    x = torch.randn(32)
    t = database_with_random_tensor_data.encoders['torch.float32[32]']
    r['x'] = t(x)
    database_with_random_tensor_data.execute(
        Collection(name='documents').replace_one(
            {'_id': r['_id']},
            r,
        )
    )


def test_dataset(database_with_random_tensor_data):
    d = Dataset(
        identifier='test_dataset',
        select=Collection(name='documents').find({'_fold': 'valid'}),
    )
    database_with_random_tensor_data.add(d)
    assert database_with_random_tensor_data.show('dataset') == ['test_dataset']
    dataset = database_with_random_tensor_data.load('dataset', 'test_dataset')
    assert len(dataset.data) == len(
        list(database_with_random_tensor_data.execute(dataset.select))
    )
