import pytest

try:
    import torch

    from superduperdb.ext.torch.encoder import tensor
    from superduperdb.ext.torch.model import TorchModel
except ImportError:
    torch = None


from superduperdb.backends.mongodb.query import Collection
from superduperdb.base.document import Document
from superduperdb.base.exceptions import ComponentInUseError, ComponentInUseWarning
from superduperdb.components.dataset import Dataset
from superduperdb.components.encoder import Encoder
from superduperdb.components.listener import Listener

n_data_points = 250


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_create_component(local_empty_data_layer):
    local_empty_data_layer.add(
        TorchModel(object=torch.nn.Linear(16, 32), identifier='my-test-module')
    )
    model = local_empty_data_layer.models['my-test-module']
    assert 'my-test-module' in local_empty_data_layer.show('model')
    print(model)
    output = model.predict(torch.randn(16), one=True)
    assert output.shape[0] == 32


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_update_component(local_empty_data_layer):
    local_empty_data_layer.add(
        TorchModel(object=torch.nn.Linear(16, 32), identifier='my-test-module')
    )
    m = TorchModel(object=torch.nn.Linear(16, 32), identifier='my-test-module')
    local_empty_data_layer.add(m)
    assert local_empty_data_layer.show('model', 'my-test-module') == [0, 1]
    local_empty_data_layer.add(m)
    assert local_empty_data_layer.show('model', 'my-test-module') == [0, 1]

    n = local_empty_data_layer.models[m.identifier]
    local_empty_data_layer.add(n)
    assert local_empty_data_layer.show('model', 'my-test-module') == [0, 1]


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_compound_component(local_empty_data_layer):
    t = tensor(torch.float, shape=(32,))

    m = TorchModel(
        object=torch.nn.Linear(16, 32),
        identifier='my-test-module',
        encoder=t,
    )

    local_empty_data_layer.add(m)
    assert 'torch.float32[32]' in local_empty_data_layer.show('encoder')
    assert 'my-test-module' in local_empty_data_layer.show('model')
    assert local_empty_data_layer.show('model', 'my-test-module') == [0]

    local_empty_data_layer.add(m)
    assert local_empty_data_layer.show('model', 'my-test-module') == [0]
    assert local_empty_data_layer.show('encoder', 'torch.float32[32]') == [0]

    local_empty_data_layer.add(
        TorchModel(
            object=torch.nn.Linear(16, 32),
            identifier='my-test-module',
            encoder=t,
        )
    )
    assert local_empty_data_layer.show('model', 'my-test-module') == [0, 1]
    assert local_empty_data_layer.show('encoder', 'torch.float32[32]') == [0]

    m = local_empty_data_layer.load(type_id='model', identifier='my-test-module')
    assert isinstance(m.encoder, Encoder)

    with pytest.raises(ComponentInUseError):
        local_empty_data_layer.remove('encoder', 'torch.float32[32]')

    with pytest.warns(ComponentInUseWarning):
        local_empty_data_layer.remove('encoder', 'torch.float32[32]', force=True)

    local_empty_data_layer.remove('model', 'my-test-module', force=True)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_reload_dataset(local_data_layer):
    from superduperdb.components.dataset import Dataset

    d = Dataset(
        identifier='my_valid',
        select=Collection('documents').find({'_fold': 'valid'}),
        sample_size=100,
    )
    local_data_layer.add(d)
    new_d = local_data_layer.load('dataset', 'my_valid')
    assert new_d.sample_size == 100


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize(
    'local_data_layer', [{'add_vector_index': False}], indirect=True
)
def test_listener(local_data_layer):
    local_data_layer.add(
        Listener(
            model='linear_a',
            select=Collection('documents').find(),
            key='x',
        ),
    )
    r = local_data_layer.execute(Collection('documents').find_one())
    assert 'linear_a' in r['_outputs']['x']

    t = local_data_layer.encoders['torch.float32[32]']

    local_data_layer.execute(
        Collection('documents').insert_many(
            [Document({'x': t(torch.randn(32)), 'update': True}) for _ in range(5)]
        )
    )

    r = local_data_layer.execute(Collection('documents').find_one({'update': True}))
    assert 'linear_a' in r['_outputs']['x']

    local_data_layer.add(
        Listener(
            model='linear_b',
            select=Collection('documents').find().featurize({'x': 'linear_a'}),
            key='x',
        )
    )
    r = local_data_layer.execute(Collection('documents').find_one())
    assert 'linear_b' in r['_outputs']['x']


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize(
    'local_data_layer', [{'add_vector_index': False}], indirect=True
)
def test_predict(local_data_layer):
    t = local_data_layer.encoders['torch.float32[32]']
    local_data_layer.predict('linear_a', Document(t(torch.randn(32))))


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize(
    'local_data_layer', [{'add_vector_index': False}], indirect=True
)
def test_dataset(local_data_layer):
    d = Dataset(
        identifier='test_dataset',
        select=Collection('documents').find({'_fold': 'valid'}),
    )
    local_data_layer.add(d)
    assert local_data_layer.show('dataset') == ['test_dataset']
    dataset = local_data_layer.load('dataset', 'test_dataset')
    assert len(dataset.data) == len(list(local_data_layer.execute(dataset.select)))
