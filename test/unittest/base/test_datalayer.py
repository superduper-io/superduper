from typing import Optional

import pytest

try:
    import torch

    from superduperdb.ext.torch.encoder import tensor
    from superduperdb.ext.torch.model import TorchModel
except ImportError:
    torch = None


import dataclasses as dc
from unittest.mock import MagicMock, patch

from superduperdb.backends.mongodb.query import Collection
from superduperdb.base.artifact import Artifact, ArtifactSavingError
from superduperdb.base.datalayer import Datalayer
from superduperdb.base.document import Document
from superduperdb.base.exceptions import ComponentInUseError, ComponentInUseWarning
from superduperdb.components.component import Component
from superduperdb.components.dataset import Dataset
from superduperdb.components.encoder import Encoder
from superduperdb.components.listener import Listener
from superduperdb.components.model import Model

n_data_points = 250


@dc.dataclass
class TestComponent(Component):
    identifier: str
    version: Optional[int] = None
    type_id: str = 'test-component'
    is_on_create: bool = False
    is_on_load: bool = False
    is_schedule_jobs: bool = False
    check_clean_up: bool = False
    child: Optional['TestComponent'] = None
    artifact: Optional[Artifact] = None

    def on_create(self, db):
        self.is_on_create = True

    def on_load(self, db):
        self.is_on_load = True

    def schedule_jobs(self, db, dependencies):
        self.is_schedule_jobs = True
        return []

    def cleanup(self, db):
        if self.check_clean_up:
            raise Exception('cleanup')

    @property
    def artifact_attributes(self):
        return ['artifact'] if self.artifact else []

    @property
    def child_components(self):
        return [('child', 'test-component')] if self.child else []


def add_fake_model(db: Datalayer):
    model = Model(object=lambda x: str(x), identifier='fake_model')
    db.add(model)
    db.add(
        Listener(
            model='fake_model',
            select=Collection('documents').find(),
            key='x',
        ),
    )


def test_add_version(local_empty_db):
    # Check the component functions are called
    component = TestComponent(identifier='test')
    local_empty_db.add(component)
    assert component.is_on_create is True
    assert component.is_on_load is True
    assert component.is_schedule_jobs is True
    assert component.version == 0
    assert local_empty_db.show('test-component', 'test') == [0]

    # Test the component saves the data correctly
    component_loaded = local_empty_db.load('test-component', 'test')

    original_serialized = component.serialized[0]
    saved_serialized = component_loaded.serialized[0]
    assert original_serialized['cls'] == saved_serialized['cls']
    assert original_serialized['module'] == saved_serialized['module']
    assert original_serialized['type_id'] == saved_serialized['type_id']
    assert original_serialized['identifier'] == saved_serialized['identifier']

    # Check duplicate components are not added
    local_empty_db.add(component)
    component_loaded = local_empty_db.load('test-component', 'test')
    assert component_loaded.version == 0
    assert local_empty_db.show('test-component', 'test') == [0]

    # Check the version is incremented
    component = TestComponent(identifier='test')
    local_empty_db.add(component)
    assert component.version == 1
    assert local_empty_db.show('test-component', 'test') == [0, 1]

    component = TestComponent(identifier='test')
    local_empty_db.add(component)
    assert component.version == 2
    assert local_empty_db.show('test-component', 'test') == [0, 1, 2]


def test_add_compenent_with_bad_artifact(local_empty_db):
    artifact = Artifact({'data': lambda x: x}, serializer='pickle')
    component = TestComponent(identifier='test', artifact=artifact)
    with pytest.raises(ArtifactSavingError):
        local_empty_db.add(component)


def test_add_artifact_auto_replace(local_empty_db):
    # Check artifact is automatically replaced to metadata
    artifact = Artifact({'data': 1})
    component = TestComponent(identifier='test', artifact=artifact)
    with patch.object(local_empty_db.metadata, 'create_component') as create_component:
        local_empty_db.add(component)
        serialized = create_component.call_args[0][0]
        assert serialized['dict']['artifact']['sha1'] == artifact.sha1


def test_add_child(local_empty_db):
    child_component = TestComponent(identifier='child')
    component = TestComponent(identifier='test', child=child_component)

    local_empty_db.add(component)
    assert local_empty_db.show('test-component', 'test') == [0]
    assert local_empty_db.show('test-component', 'child') == [0]

    parents = local_empty_db.metadata.get_component_version_parents(
        child_component.unique_id
    )
    assert parents == [component.unique_id]

    component_2 = TestComponent(identifier='test-2', child='child-2')
    with pytest.raises(FileNotFoundError):
        local_empty_db.add(component_2)

    child_component_2 = TestComponent(identifier='child-2')
    local_empty_db.add(child_component_2)
    local_empty_db.add(component_2)
    assert local_empty_db.show('test-component', 'test-2') == [0]
    assert local_empty_db.show('test-component', 'child-2') == [0]

    parents = local_empty_db.metadata.get_component_version_parents(
        child_component_2.unique_id
    )
    assert parents == [component_2.unique_id]


def test_add(local_empty_db):
    component = TestComponent(identifier='test')
    local_empty_db.add(component)
    assert local_empty_db.show('test-component', 'test') == [0]

    local_empty_db.add(
        [
            TestComponent(identifier='test_list_1'),
            TestComponent(identifier='test_list_2'),
        ]
    )
    assert local_empty_db.show('test-component', 'test_list_1') == [0]
    assert local_empty_db.show('test-component', 'test_list_2') == [0]

    with pytest.raises(ValueError):
        local_empty_db.add('test')


def test_remove_component_version(local_empty_db):
    local_empty_db.add(
        [
            TestComponent(identifier='test', version=0),
            TestComponent(identifier='test', version=1),
        ]
    )
    assert local_empty_db.show('test-component', 'test') == [0, 1]

    # Don't remove if not confirmed
    with patch('click.confirm', return_value=False):
        local_empty_db._remove_component_version('test-component', 'test', 0)
        assert local_empty_db.show('test-component', 'test') == [0, 1]

    # Remove if confirmed
    with patch('click.confirm', return_value=True):
        local_empty_db._remove_component_version('test-component', 'test', 0)
        assert local_empty_db.show('test-component', 'test') == [1]

    # Remove force
    local_empty_db._remove_component_version('test-component', 'test', 1, force=True)
    assert local_empty_db.show('test-component', 'test') == []


def test_remove_component_with_parent(local_empty_db):
    # Can not remove the child component if the parent exists
    local_empty_db.add(
        TestComponent(
            identifier='test_3_parent',
            version=0,
            child=TestComponent(identifier='test_3_child', version=0),
        )
    )
    with pytest.raises(Exception) as e:
        local_empty_db._remove_component_version('test-component', 'test_3_child', 0)
        assert 'test_3_parent' in str(e.value)


def test_remove_component_with_clean_up(local_empty_db):
    # Test clean up
    component_clean_up = TestComponent(
        identifier='test_clean_up', version=0, check_clean_up=True
    )
    local_empty_db.add(component_clean_up)
    with pytest.raises(Exception) as e:
        local_empty_db._remove_component_version(
            'test-component', 'test_clean_up', 0, force=True
        )
        assert 'test_clean_up' in str(e.value)


def test_remove_component_from_data_layer_dict(local_empty_db):
    # Test component is deleted from datalayer
    test_encoder = Encoder(identifier='test_encoder', version=0)
    local_empty_db.add(test_encoder)
    local_empty_db._remove_component_version('encoder', 'test_encoder', 0, force=True)
    with pytest.raises(FileNotFoundError):
        local_empty_db.encoders['test_encoder']


def test_remove_component_with_artifact(local_empty_db):
    # Test artifact is deleted from artifact store
    component_with_artifact = TestComponent(
        identifier='test_with_artifact', version=0, artifact=Artifact({'test': 'test'})
    )
    local_empty_db.add(component_with_artifact)
    info_with_artifact = local_empty_db.metadata.get_component(
        'test-component', 'test_with_artifact', 0
    )
    artifact_file_id = info_with_artifact['dict']['artifact']['file_id']
    with patch.object(local_empty_db.artifact_store, 'delete') as mock_delete:
        local_empty_db._remove_component_version(
            'test-component', 'test_with_artifact', 0, force=True
        )
        mock_delete.assert_called_once_with(artifact_file_id)


def test_remove_one_version(local_empty_db):
    local_empty_db.add(
        [
            TestComponent(identifier='test', version=0),
            TestComponent(identifier='test', version=1),
        ]
    )

    # Only remove the version
    local_empty_db.remove('test-component', 'test', 1, force=True)
    assert local_empty_db.show('test-component', 'test') == [0]


def test_remove_multi_version(local_empty_db):
    local_empty_db.add(
        [
            TestComponent(identifier='test', version=0),
            TestComponent(identifier='test', version=1),
            TestComponent(identifier='test', version=2),
        ]
    )

    local_empty_db.remove('test-component', 'test', force=True)
    assert local_empty_db.show('test-component', 'test') == []


def test_remove_not_exist_component(local_empty_db):
    with pytest.raises(FileNotFoundError) as e:
        local_empty_db.remove('test-component', 'test', 0, force=True)
        assert 'test' in str(e.value)

    local_empty_db.remove('test-component', 'test', force=True)


def test_show(local_empty_db):
    local_empty_db.add(
        [
            TestComponent(identifier='a1', version=0),
            TestComponent(identifier='a2', version=1),
            TestComponent(identifier='a3', version=2),
            TestComponent(identifier='b', version=0),
            TestComponent(identifier='b', version=1),
            TestComponent(identifier='b', version=2),
            Encoder(identifier='c1', version=0),
            Encoder(identifier='c2', version=0),
        ]
    )

    with pytest.raises(ValueError) as e:
        local_empty_db.show('test-component', version=1)
        assert 'test-component' in str(e.value) and '1' in str(e.value)

    assert sorted(local_empty_db.show('test-component')) == ['a1', 'a2', 'a3', 'b']
    assert sorted(local_empty_db.show('encoder')) == ['c1', 'c2']

    assert sorted(local_empty_db.show('test-component', 'a1')) == [0]
    assert sorted(local_empty_db.show('test-component', 'b')) == [0, 1, 2]

    # Test get specific version
    info = local_empty_db.show('test-component', 'b', 1)
    assert isinstance(info, dict)
    assert info['version'] == 1
    assert info['dict']['identifier'] == 'b'
    assert info['cls'] == 'TestComponent'

    # Test get last version
    assert local_empty_db.show('test-component', 'b', -1)['version'] == 2


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_predict(local_empty_db):
    models = [
        TorchModel(object=torch.nn.Linear(16, 2), identifier='model1'),
        TorchModel(object=torch.nn.Linear(16, 3), identifier='model2'),
        TorchModel(
            object=torch.nn.Linear(16, 3),
            identifier='model3',
            encoder=Encoder(
                identifier='test-encoder',
                version=0,
                encoder=lambda x: torch.argmax(x, dim=1),
            ),
        ),
    ]
    local_empty_db.add(models)

    # test model selection
    x = torch.randn(4, 16)
    assert local_empty_db.predict('model1', x)[0].content.shape == torch.Size([4, 2])
    assert local_empty_db.predict('model2', x)[0].content.shape == torch.Size([4, 3])

    # test encoder
    result = local_empty_db.predict('model3', torch.randn(4, 16))[0].content.encode()
    assert result['_content']['bytes'].shape == torch.Size([4])


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_predict_context(local_empty_db):
    local_empty_db.add(TorchModel(object=torch.nn.Linear(16, 2), identifier='model'))

    y, context_out = local_empty_db.predict('model', torch.randn(4, 16))
    assert not context_out

    with patch.object(local_empty_db, '_get_context') as mock_get_context:
        mock_get_context.return_value = [
            torch.randn(4, 2),
            torch.randn(4, 3),
        ]
        y, context_out = local_empty_db.predict(
            'model', torch.randn(4, 16), context_select=True
        )
        assert context_out[0].content.shape == torch.Size([4, 2])
        assert context_out[1].content.shape == torch.Size([4, 3])


def test_get_context(local_empty_db):
    from superduperdb.backends.base.query import Select

    fake_contexts = [Document(content={'text': f'hello world {i}'}) for i in range(10)]

    model = Model(object=lambda x: x, identifier='model', takes_context=True)
    context_select = MagicMock(spec=Select)
    context_select.execute.return_value = fake_contexts

    # Test get_context without context_key
    return_contexts = local_empty_db._get_context(
        model, context_select, context_key=None
    )
    assert return_contexts == [{'text': f'hello world {i}'} for i in range(10)]

    # Test get context without context
    return_contexts = local_empty_db._get_context(
        model, context_select, context_key='text'
    )
    assert return_contexts == [f'hello world {i}' for i in range(10)]

    # Testing models that cannot accept context
    model = Model(object=lambda x: x, identifier='model', takes_context=False)
    with pytest.raises(AssertionError):
        local_empty_db._get_context(model, context_select, context_key=None)


def test_load(local_empty_db):
    local_empty_db.add(
        [
            Encoder(identifier='e1', version=0),
            Encoder(identifier='e2', version=0),
            Model(object=lambda x: x, identifier='m1', version=0),
            Model(object=lambda x: x, identifier='m1', version=1),
            Model(object=lambda x: x, identifier='m2', version=0),
        ]
    )

    # Test load fails
    # error version
    with pytest.raises(Exception):
        local_empty_db.load('encoder', 'e1', version=1)

    # error identifier
    with pytest.raises(Exception):
        local_empty_db.load('encoder', 'm1')

    info = local_empty_db.load('encoder', 'e1', info_only=True)
    assert isinstance(info, dict)

    encoder = local_empty_db.load('encoder', 'e1')
    assert isinstance(encoder, Encoder)

    assert 'e1' in local_empty_db.encoders


def test_insert(local_empty_db):
    add_fake_model(local_empty_db)
    inserted_ids, _ = local_empty_db.insert(
        Collection('documents').insert_many(
            [Document({'x': i, 'update': True}) for i in range(5)]
        )
    )
    assert len(inserted_ids) == 5

    new_docs = list(
        local_empty_db.execute(
            Collection('documents').find().select_using_ids(inserted_ids)
        )
    )
    result = [doc.outputs('x', 'fake_model') for doc in new_docs]
    assert sorted(result) == ['0', '1', '2', '3', '4']


def test_update(local_empty_db):
    add_fake_model(local_empty_db)
    local_empty_db.insert(
        Collection('documents').insert_many(
            [Document({'x': i, 'update': True}) for i in range(5)]
        )
    )
    updated_ids, _ = local_empty_db.update(
        Collection('documents').update_many({}, Document({'$set': {'x': 100}}))
    )
    assert len(updated_ids) == 5
    new_docs = list(
        local_empty_db.execute(
            Collection('documents').find().select_using_ids(updated_ids)
        )
    )
    result = [doc.outputs('x', 'fake_model') for doc in new_docs]
    assert result == ['100'] * 5


def test_delete(local_empty_db):
    local_empty_db.insert(
        Collection('documents').insert_many(
            [Document({'x': i, 'update': True}) for i in range(5)]
        )
    )
    local_empty_db.delete(Collection('documents').delete_one({}))
    new_docs = list(local_empty_db.execute(Collection('documents').find()))
    assert len(new_docs) == 4


def test_replace(local_empty_db):
    model = Model(object=lambda x: x + 1, identifier='m', version=0)
    with pytest.raises(Exception):
        local_empty_db.replace(model)

    local_empty_db.replace(model, upsert=True)

    assert local_empty_db.load('model', 'm').predict([1]) == [2]

    # replace the 0 version of the model
    new_model = Model(object=lambda x: x + 2, identifier='m', version=0)
    local_empty_db.replace(new_model)
    assert local_empty_db.load('model', 'm').predict([1]) == [3]

    # replace the last version of the model
    new_model = Model(object=lambda x: x + 3, identifier='m')
    local_empty_db.replace(new_model)
    assert local_empty_db.load('model', 'm').predict([1]) == [4]


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_compound_component(local_empty_db):
    t = tensor(torch.float, shape=(32,))

    m = TorchModel(
        object=torch.nn.Linear(16, 32),
        identifier='my-test-module',
        encoder=t,
    )

    local_empty_db.add(m)
    assert 'torch.float32[32]' in local_empty_db.show('encoder')
    assert 'my-test-module' in local_empty_db.show('model')
    assert local_empty_db.show('model', 'my-test-module') == [0]

    local_empty_db.add(m)
    assert local_empty_db.show('model', 'my-test-module') == [0]
    assert local_empty_db.show('encoder', 'torch.float32[32]') == [0]

    local_empty_db.add(
        TorchModel(
            object=torch.nn.Linear(16, 32),
            identifier='my-test-module',
            encoder=t,
        )
    )
    assert local_empty_db.show('model', 'my-test-module') == [0, 1]
    assert local_empty_db.show('encoder', 'torch.float32[32]') == [0]

    m = local_empty_db.load(type_id='model', identifier='my-test-module')
    assert isinstance(m.encoder, Encoder)

    with pytest.raises(ComponentInUseError):
        local_empty_db.remove('encoder', 'torch.float32[32]')

    with pytest.warns(ComponentInUseWarning):
        local_empty_db.remove('encoder', 'torch.float32[32]', force=True)

    local_empty_db.remove('model', 'my-test-module', force=True)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_reload_dataset(local_db):
    from superduperdb.components.dataset import Dataset

    d = Dataset(
        identifier='my_valid',
        select=Collection('documents').find({'_fold': 'valid'}),
        sample_size=100,
    )
    local_db.add(d)
    new_d = local_db.load('dataset', 'my_valid')
    assert new_d.sample_size == 100


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize('local_db', [{'add_vector_index': False}], indirect=True)
def test_dataset(local_db):
    d = Dataset(
        identifier='test_dataset',
        select=Collection('documents').find({'_fold': 'valid'}),
    )
    local_db.add(d)
    assert local_db.show('dataset') == ['test_dataset']
    dataset = local_db.load('dataset', 'test_dataset')
    assert len(dataset.data) == len(list(local_db.execute(dataset.select)))


# TODO: add UT for task workflow
