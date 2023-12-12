import time
from typing import Optional

import pytest

try:
    import torch

    from superduperdb.ext.torch.encoder import tensor
    from superduperdb.ext.torch.model import TorchModel
except ImportError:
    torch = None


import dataclasses as dc
from test.db_config import DBConfig
from unittest.mock import MagicMock, patch

from superduperdb.backends.ibis.field_types import dtype
from superduperdb.backends.ibis.query import Table
from superduperdb.backends.mongodb.data_backend import MongoDataBackend
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
from superduperdb.components.schema import Schema

n_data_points = 250


@dc.dataclass(kw_only=True)
class TestComponent(Component):
    version: Optional[int] = None
    type_id: str = 'test-component'
    is_on_create: bool = False
    is_after_create: bool = False
    is_schedule_jobs: bool = False
    check_clean_up: bool = False
    child: Optional['TestComponent'] = None
    artifact: Optional[Artifact] = None

    def pre_create(self, db):
        self.is_on_create = True

    def post_create(self, db):
        self.is_after_create = True

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
    model = Model(
        object=lambda x: str(x),
        identifier='fake_model',
        encoder=Encoder(identifier='base'),
    )
    db.add(model)
    if isinstance(db.databackend, MongoDataBackend):
        select = Collection('documents').find()
    else:
        schema = Schema(
            identifier='documents',
            fields={
                'id': dtype('str'),
                'x': dtype('int'),
            },
        )
        t = Table(identifier='documents', schema=schema)
        db.add(t)
        select = db.load('table', 'documents').to_query().select('id', 'x')
    db.add(
        Listener(
            model='fake_model',
            select=select,
            key='x',
        ),
    )


@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_add_version(db: Datalayer):
    # Check the component functions are called
    component = TestComponent(identifier='test')
    db.add(component)
    assert component.is_on_create is True
    assert component.is_after_create is True
    assert component.is_schedule_jobs is True
    assert component.version == 0
    assert db.show('test-component', 'test') == [0]

    # Test the component saves the data correctly
    component_loaded = db.load('test-component', 'test')

    original_serialized = component.serialized[0]
    saved_serialized = component_loaded.serialized[0]
    assert original_serialized['cls'] == saved_serialized['cls']
    assert original_serialized['module'] == saved_serialized['module']
    assert original_serialized['type_id'] == saved_serialized['type_id']
    assert original_serialized['identifier'] == saved_serialized['identifier']

    # Check duplicate components are not added
    db.add(component)
    component_loaded = db.load('test-component', 'test')
    assert component_loaded.version == 0
    assert db.show('test-component', 'test') == [0]

    # Check the version is incremented
    component = TestComponent(identifier='test')
    db.add(component)
    assert component.version == 1
    assert db.show('test-component', 'test') == [0, 1]

    component = TestComponent(identifier='test')
    db.add(component)
    assert component.version == 2
    assert db.show('test-component', 'test') == [0, 1, 2]


@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_add_component_with_bad_artifact(db):
    artifact = Artifact({'data': lambda x: x}, serializer='pickle')
    component = TestComponent(identifier='test', artifact=artifact)
    with pytest.raises(ArtifactSavingError):
        db.add(component)


@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_add_artifact_auto_replace(db):
    # Check artifact is automatically replaced to metadata
    artifact = Artifact({'data': 1})
    component = TestComponent(identifier='test', artifact=artifact)
    with patch.object(db.metadata, 'create_component') as create_component:
        db.add(component)
        serialized = create_component.call_args[0][0]
        assert serialized['dict']['artifact']['sha1'] == artifact.sha1(db.serializers)


@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_add_child(db):
    child_component = TestComponent(identifier='child')
    component = TestComponent(identifier='test', child=child_component)

    db.add(component)
    assert db.show('test-component', 'test') == [0]
    assert db.show('test-component', 'child') == [0]

    parents = db.metadata.get_component_version_parents(child_component.unique_id)
    assert parents == [component.unique_id]

    component_2 = TestComponent(identifier='test-2', child='child-2')
    with pytest.raises(FileNotFoundError):
        db.add(component_2)

    child_component_2 = TestComponent(identifier='child-2')
    db.add(child_component_2)
    component_3 = TestComponent(identifier='test-3', child='child-2')
    db.add(component_3)
    assert db.show('test-component', 'test-3') == [0]
    assert db.show('test-component', 'child-2') == [0]

    parents = db.metadata.get_component_version_parents(child_component_2.unique_id)
    assert parents == [component_3.unique_id]


@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_add(db):
    component = TestComponent(identifier='test')
    db.add(component)
    assert db.show('test-component', 'test') == [0]

    db.add(
        [
            TestComponent(identifier='test_list_1'),
            TestComponent(identifier='test_list_2'),
        ]
    )
    assert db.show('test-component', 'test_list_1') == [0]
    assert db.show('test-component', 'test_list_2') == [0]

    with pytest.raises(ValueError):
        db.add('test')


@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_remove_component_version(db):
    db.add(
        [
            TestComponent(identifier='test', version=0),
            TestComponent(identifier='test', version=1),
        ]
    )
    assert db.show('test-component', 'test') == [0, 1]

    # Don't remove if not confirmed
    with patch('click.confirm', return_value=False):
        db._remove_component_version('test-component', 'test', 0)
        assert db.show('test-component', 'test') == [0, 1]

    # Remove if confirmed
    with patch('click.confirm', return_value=True):
        db._remove_component_version('test-component', 'test', 0)
        # Wait for the db to update
        time.sleep(0.1)
        assert db.show('test-component', 'test') == [1]

    # Remove force
    db._remove_component_version('test-component', 'test', 1, force=True)
    # Wait for the db to update
    time.sleep(0.1)
    assert db.show('test-component', 'test') == []


@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_remove_component_with_parent(db):
    # Can not remove the child component if the parent exists
    db.add(
        TestComponent(
            identifier='test_3_parent',
            version=0,
            child=TestComponent(identifier='test_3_child', version=0),
        )
    )

    with pytest.raises(Exception) as e:
        db._remove_component_version('test-component', 'test_3_child', 0)
    assert 'is involved in other components' in str(e)


@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_remove_component_with_clean_up(db):
    # Test clean up
    component_clean_up = TestComponent(
        identifier='test_clean_up', version=0, check_clean_up=True
    )
    db.add(component_clean_up)
    with pytest.raises(Exception) as e:
        db._remove_component_version('test-component', 'test_clean_up', 0, force=True)
    assert 'cleanup' in str(e)


@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_remove_component_from_data_layer_dict(db):
    # Test component is deleted from datalayer
    test_encoder = Encoder(identifier='test_encoder')
    db.add(test_encoder)
    db._remove_component_version('encoder', 'test_encoder', 0, force=True)
    with pytest.raises(FileNotFoundError):
        db.encoders['test_encoder']


@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_remove_component_with_artifact(db):
    # Test artifact is deleted from artifact store
    component_with_artifact = TestComponent(
        identifier='test_with_artifact', version=0, artifact=Artifact({'test': 'test'})
    )
    db.add(component_with_artifact)
    info_with_artifact = db.metadata.get_component(
        'test-component', 'test_with_artifact', 0
    )
    artifact_file_id = info_with_artifact['dict']['artifact']['file_id']
    with patch.object(db.artifact_store, 'delete') as mock_delete:
        db._remove_component_version(
            'test-component', 'test_with_artifact', 0, force=True
        )
        mock_delete.assert_called_once_with(artifact_file_id)


@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_remove_one_version(db):
    db.add(
        [
            TestComponent(identifier='test', version=0),
            TestComponent(identifier='test', version=1),
        ]
    )

    # Only remove the version
    db.remove('test-component', 'test', 1, force=True)
    # Wait for the db to update
    time.sleep(0.1)
    assert db.show('test-component', 'test') == [0]


@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_remove_multi_version(db):
    db.add(
        [
            TestComponent(identifier='test', version=0),
            TestComponent(identifier='test', version=1),
            TestComponent(identifier='test', version=2),
        ]
    )

    db.remove('test-component', 'test', force=True)
    # Wait for the db to update
    time.sleep(0.1)
    assert db.show('test-component', 'test') == []


@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_remove_not_exist_component(db):
    with pytest.raises(FileNotFoundError) as e:
        db.remove('test-component', 'test', 0, force=True)
    assert 'test' in str(e)

    db.remove('test-component', 'test', force=True)


@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_show(db):
    db.add(
        [
            TestComponent(identifier='a1'),
            TestComponent(identifier='a2'),
            TestComponent(identifier='a3'),
            TestComponent(identifier='b'),
            TestComponent(identifier='b'),
            TestComponent(identifier='b'),
            Encoder(identifier='c1'),
            Encoder(identifier='c2'),
        ]
    )

    with pytest.raises(ValueError) as e:
        db.show('test-component', version=1)
    assert 'None' in str(e) and '1' in str(e)

    assert sorted(db.show('test-component')) == ['a1', 'a2', 'a3', 'b']
    assert sorted(db.show('encoder')) == ['c1', 'c2']

    assert sorted(db.show('test-component', 'a1')) == [0]
    assert sorted(db.show('test-component', 'b')) == [0, 1, 2]

    # Test get specific version
    info = db.show('test-component', 'b', 1)
    assert isinstance(info, dict)
    assert info['version'] == 1
    assert info['dict']['identifier'] == 'b'
    assert info['cls'] == 'TestComponent'

    # Test get last version
    assert db.show('test-component', 'b', -1)['version'] == 2


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_predict(db: Datalayer):
    models = [
        TorchModel(
            object=torch.nn.Linear(16, 2),
            identifier='model1',
            encoder=tensor(torch.float32, shape=(4, 2)),
        ),
        TorchModel(
            object=torch.nn.Linear(16, 3),
            identifier='model2',
            encoder=tensor(torch.float32, shape=(4, 3)),
        ),
        TorchModel(
            object=torch.nn.Linear(16, 3),
            identifier='model3',
            encoder=Encoder(
                identifier='test-encoder',
                encoder=lambda x: torch.argmax(x, dim=1),
            ),
        ),
    ]
    db.add(models)

    # test model selection
    x = torch.randn(4, 16)
    assert db.predict('model1', x)[0].content.x.shape == torch.Size([4, 2])
    assert db.predict('model2', x)[0].content.x.shape == torch.Size([4, 3])

    # test encoder
    result = db.predict('model3', torch.randn(4, 16))[0].content.encode()
    assert result['_content']['bytes'].shape == torch.Size([4])


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_predict_context(db: Datalayer):
    db.add(
        TorchModel(
            object=torch.nn.Linear(16, 2),
            identifier='model',
            encoder=tensor(torch.float32, shape=(4, 2)),
        )
    )

    y, context_out = db.predict('model', torch.randn(4, 16))
    assert not context_out

    with patch.object(db, '_get_context') as mock_get_context:
        mock_get_context.return_value = (
            [None, None],
            [Document(torch.randn(4, 2)), Document(torch.randn(4, 3))],
        )
        y, context_out = db.predict(
            'model',
            torch.randn(4, 16),
            context_select=Collection('context_collection').find({}),
        )
        assert context_out[0].content.shape == torch.Size([4, 2])
        assert context_out[1].content.shape == torch.Size([4, 3])


@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_get_context(db):
    from superduperdb.backends.base.query import Select

    fake_contexts = [Document(content={'text': f'hello world {i}'}) for i in range(10)]

    model = Model(object=lambda x: x, identifier='model', takes_context=True)
    context_select = MagicMock(spec=Select)
    context_select.variables = []
    context_select.execute.return_value = fake_contexts

    # Test get_context without context_key
    return_contexts, _ = db._get_context(model, context_select, context_key=None)
    assert return_contexts == [{'text': f'hello world {i}'} for i in range(10)]

    # Test get context without context
    return_contexts, _ = db._get_context(model, context_select, context_key='text')
    assert return_contexts == [f'hello world {i}' for i in range(10)]

    # Testing models that cannot accept context
    model = Model(object=lambda x: x, identifier='model', takes_context=False)
    with pytest.raises(AssertionError):
        db._get_context(model, context_select, context_key=None)


@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_load(db):
    m1 = Model(object=lambda x: x, identifier='m1', encoder=dtype('int32'))
    db.add(
        [
            Encoder(identifier='e1'),
            Encoder(identifier='e2'),
            m1,
            Model(object=lambda x: x, identifier='m1', encoder=dtype('int32')),
            m1,
        ]
    )

    # Test load fails
    # error version
    with pytest.raises(Exception):
        db.load('encoder', 'e1', version=1)

    # error identifier
    with pytest.raises(Exception):
        db.load('encoder', 'm1')

    info = db.load('encoder', 'e1', info_only=True)
    assert isinstance(info, dict)

    encoder = db.load('encoder', 'e1')
    assert isinstance(encoder, Encoder)

    assert 'e1' in db.encoders


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_insert_mongo_db(db):
    add_fake_model(db)
    inserted_ids, _ = db.insert(
        Collection('documents').insert_many(
            [Document({'x': i, 'update': True}) for i in range(5)]
        )
    )
    assert len(inserted_ids) == 5

    new_docs = list(
        db.execute(Collection('documents').find().select_using_ids(inserted_ids))
    )
    result = [doc.outputs('x', 'fake_model') for doc in new_docs]
    assert sorted(result) == ['0', '1', '2', '3', '4']


@pytest.mark.parametrize("db", [DBConfig.sqldb_empty], indirect=True)
def test_insert_sql_db(db):
    add_fake_model(db)
    table = db.load('table', 'documents')
    inserted_ids, _ = db.insert(
        table.insert([Document({'id': str(i), 'x': i}) for i in range(5)])
    )
    assert len(inserted_ids) == 5

    q = table.select('id', 'x').outputs(x='fake_model/0')
    new_docs = db.execute(q)
    new_docs = list(new_docs)
    # new_docs = list(db.execute(table.outputs(x='fake_model/0')))

    result = [doc.unpack()['_outputs.x.fake_model.0'] for doc in new_docs]
    assert sorted(result) == ['0', '1', '2', '3', '4']


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_update_db(db):
    # TODO: test update sql db after the update method is implemented
    add_fake_model(db)
    db.insert(
        Collection('documents').insert_many(
            [Document({'x': i, 'update': True}) for i in range(5)]
        )
    )
    updated_ids, _ = db.update(
        Collection('documents').update_many({}, Document({'$set': {'x': 100}}))
    )
    assert len(updated_ids) == 5
    new_docs = list(
        db.execute(Collection('documents').find().select_using_ids(updated_ids))
    )
    result = [doc.outputs('x', 'fake_model') for doc in new_docs]
    assert result == ['100'] * 5


@pytest.mark.parametrize(
    "db",
    [
        (DBConfig.mongodb_data, {'n_data': 6}),
    ],
    indirect=True,
)
def test_delete(db):
    # TODO: add sqldb test after the delete method is implemented
    db.delete(Collection('documents').delete_one({}))
    new_docs = list(db.execute(Collection('documents').find()))
    assert len(new_docs) == 5


@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_replace(db):
    model = Model(
        object=lambda x: x + 1,
        identifier='m',
        encoder=Encoder(identifier='base'),
    )
    model.version = 0
    with pytest.raises(Exception):
        db.replace(model)

    db.replace(model, upsert=True)

    assert db.load('model', 'm').predict([1]) == [2]

    # replace the 0 version of the model
    new_model = Model(object=lambda x: x + 2, identifier='m')
    new_model.version = 0
    db.replace(new_model)
    time.sleep(0.1)
    assert db.load('model', 'm').predict([1]) == [3]

    # replace the last version of the model
    new_model = Model(object=lambda x: x + 3, identifier='m')
    db.replace(new_model)
    time.sleep(0.1)
    assert db.load('model', 'm').predict([1]) == [4]


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_compound_component(db):
    m = TorchModel(
        object=torch.nn.Linear(16, 32),
        identifier='my-test-module',
        encoder=tensor(torch.float, shape=(32,)),
    )

    db.add(m)
    assert 'torch.float32[32]' in db.show('encoder')
    assert 'my-test-module' in db.show('model')
    assert db.show('model', 'my-test-module') == [0]

    db.add(m)
    assert db.show('model', 'my-test-module') == [0]
    assert db.show('encoder', 'torch.float32[32]') == [0]

    db.add(
        TorchModel(
            object=torch.nn.Linear(16, 32),
            identifier='my-test-module',
            encoder=tensor(torch.float, shape=(32,)),
        )
    )
    assert db.show('model', 'my-test-module') == [0, 1]
    assert db.show('encoder', 'torch.float32[32]') == [0, 1]

    m = db.load(type_id='model', identifier='my-test-module')
    assert isinstance(m.encoder, Encoder)

    with pytest.raises(ComponentInUseError):
        db.remove('encoder', 'torch.float32[32]')

    with pytest.warns(ComponentInUseWarning):
        db.remove('encoder', 'torch.float32[32]', force=True)

    db.remove('model', 'my-test-module', force=True)


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize("db", [DBConfig.mongodb, DBConfig.sqldb], indirect=True)
def test_reload_dataset(db):
    from superduperdb.components.dataset import Dataset

    if isinstance(db.databackend, MongoDataBackend):
        select = Collection('documents').find({'_fold': 'valid'})
    else:
        table = db.load('table', 'documents')
        select = table.select('id', 'x', 'y', 'z').filter(table._fold == 'valid')

    d = Dataset(
        identifier='my_valid',
        select=select,
        sample_size=100,
    )
    db.add(d)
    new_d = db.load('dataset', 'my_valid')
    assert new_d.sample_size == 100


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize(
    "db",
    [
        (DBConfig.mongodb_no_vector_index, {'n_data': n_data_points}),
        (DBConfig.sqldb_no_vector_index, {'n_data': n_data_points}),
    ],
    indirect=True,
)
def test_dataset(db):
    if isinstance(db.databackend, MongoDataBackend):
        select = Collection('documents').find({'_fold': 'valid'})
    else:
        table = db.load('table', 'documents')
        select = table.select('id', 'x', 'y', 'z').filter(table._fold == 'valid')

    d = Dataset(
        identifier='test_dataset',
        select=select,
    )
    db.add(d)
    assert db.show('dataset') == ['test_dataset']
    dataset = db.load('dataset', 'test_dataset')
    assert len(dataset.data) == len(list(db.execute(dataset.select)))


# TODO: add UT for task workflow
