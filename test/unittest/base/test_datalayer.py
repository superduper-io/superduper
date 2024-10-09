import dataclasses as dc
import os
import time
from typing import Any, ClassVar, Optional, Sequence
from unittest.mock import patch

import numpy
import pytest

from superduper.base.datalayer import Datalayer
from superduper.base.document import Document
from superduper.components.component import Component
from superduper.components.dataset import Dataset
from superduper.components.datatype import (
    DataType,
    LazyArtifact,
    dill_serializer,
    pickle_decode,
    pickle_encode,
    pickle_serializer,
)
from superduper.components.listener import Listener
from superduper.components.model import Model, ObjectModel, Trainer
from superduper.components.schema import FieldType, Schema
from superduper.components.table import Table


class FakeModel(Model):
    """Fake model for testing."""

    ...


n_data_points = 250

ibis_config = os.environ.get('SUPERDUPER_CONFIG', "").endswith('ibis.yaml')
mongodb_config = os.environ.get('SUPERDUPER_CONFIG', "").endswith('mongodb.yaml')


@dc.dataclass(kw_only=True)
class TestComponent(Component):
    _artifacts: ClassVar[Sequence[str]] = (('artifact', dill_serializer),)
    version: Optional[int] = None
    type_id: str = 'test-component'
    is_on_create: bool = False
    is_after_create: bool = False
    check_clean_up: bool = False
    child: Optional['TestComponent'] = None
    artifact: Any = None

    def pre_create(self, db):
        self.is_on_create = True

    def on_create(self, db):
        self.is_after_create = True

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
    schema = Schema(
        identifier='documents',
        fields={
            'id': 'str',
            'x': 'int',
        },
    )
    t = Table(identifier='documents', schema=schema)
    db.apply(t)

    model = ObjectModel(
        object=lambda x: str(x),
        identifier='fake_model',
        example=((1,), {}),
    )
    db.apply(model)
    select = db['documents'].select()
    listener = Listener(
        identifier='listener-x',
        model=model,
        select=select,
        key='x',
    )
    db.apply(listener)
    return listener


def test_add_version(db: Datalayer):
    # Check the component functions are called
    component = TestComponent(identifier='test')
    db.apply(component)
    assert component.is_on_create is True
    # assert component.is_after_create is True
    assert component.version == 0
    assert db.show('test-component', 'test') == [0]

    # Test the component saves the data correctly
    component_loaded = db.load('test-component', 'test')

    original_serialized = component.dict().encode()
    saved_serialized = component_loaded.dict().encode()

    assert original_serialized['_path'] == saved_serialized['_path']
    assert original_serialized['type_id'] == saved_serialized['type_id']
    assert original_serialized['identifier'] == saved_serialized['identifier']

    # Check duplicate components are not added
    db.apply(component)
    component_loaded = db.load('test-component', 'test')
    assert component_loaded.version == 0
    assert db.show('test-component', 'test') == [0]

    # Check the version is incremented
    component = TestComponent(identifier='test')
    db.apply(component)
    assert component.version == 1
    assert db.show('test-component', 'test') == [0, 1]

    component = TestComponent(identifier='test')
    db.apply(component)
    assert component.version == 2
    assert db.show('test-component', 'test') == [0, 1, 2]


def test_add_component_with_bad_artifact(db):
    artifact = {'data': lambda x: x}
    component = TestComponent(
        identifier='test', artifact=artifact, artifacts={'artifact': pickle_serializer}
    )
    with pytest.raises(Exception):
        db.apply(component)


def test_add_artifact_auto_replace(db):
    # Check artifact is automatically replaced to metadata
    artifact = {'data': 1}
    component = TestComponent(identifier='test', artifact=artifact)
    db.apply(component)
    r = db.show('test-component', 'test', -1)
    assert r['artifact'].startswith('?')
    info = r['_builds'][r['artifact'][1:]]
    assert info['blob'].startswith('&')


def test_add_child(db):
    child_component = TestComponent(identifier='child')
    component = TestComponent(identifier='test', child=child_component)

    db.apply(component)
    assert db.show('test-component', 'test') == [0]
    assert db.show('test-component', 'child') == [0]

    parents = db.metadata.get_component_version_parents(child_component.uuid)
    assert parents == [component.uuid]

    child_component_2 = TestComponent(identifier='child-2')
    component_3 = TestComponent(identifier='test-3', child=child_component_2)
    db.apply(component_3)
    assert db.show('test-component', 'test-3') == [0]
    assert db.show('test-component', 'child-2') == [0]

    parents = db.metadata.get_component_version_parents(child_component_2.uuid)
    assert parents == [component_3.uuid]


def test_add(db):
    component = TestComponent(identifier='test')
    db.apply(component)
    assert db.show('test-component', 'test') == [0]

    for component in [
        TestComponent(identifier='test_list_1'),
        TestComponent(identifier='test_list_2'),
    ]:
        db.apply(component)
    assert db.show('test-component', 'test_list_1') == [0]
    assert db.show('test-component', 'test_list_2') == [0]

    with pytest.raises(ValueError):
        db.apply('test')


def test_add_with_artifact(db):
    m = ObjectModel(
        identifier='test',
        object=lambda x: x + 2,
        datatype=dill_serializer,
    )

    db.apply(m)

    m = db.load('model', m.identifier)

    assert m.object is not None

    assert isinstance(m.object, LazyArtifact)
    m.init()
    assert callable(m.object)


def test_add_table(db):
    component = Table('test', schema=Schema('test-s', fields={'field': 'str'}))
    db.apply(component)


def test_remove_component_version(db):
    for component in [
        TestComponent(identifier='test', version=0),
        TestComponent(identifier='test', version=1),
    ]:
        db.apply(component)
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


def test_remove_component_with_parent(db):
    # Can not remove the child component if the parent exists
    db.apply(
        TestComponent(
            identifier='test_3_parent',
            version=0,
            child=TestComponent(identifier='test_3_child', version=0),
        )
    )

    with pytest.raises(Exception) as e:
        db._remove_component_version('test-component', 'test_3_child', 0)
    assert 'is involved in other components' in str(e)


def test_remove_component_with_clean_up(db):
    # Test clean up
    component_clean_up = TestComponent(
        identifier='test_clean_up', version=0, check_clean_up=True
    )
    db.apply(component_clean_up)
    with pytest.raises(Exception) as e:
        db._remove_component_version('test-component', 'test_clean_up', 0, force=True)
    assert 'cleanup' in str(e)


def test_remove_component_from_data_layer_dict(db):
    # Test component is deleted from datalayer
    test_datatype = DataType(identifier='test_datatype')
    db.apply(test_datatype)
    db._remove_component_version('datatype', 'test_datatype', 0, force=True)
    with pytest.raises(FileNotFoundError):
        db.load('datatype', 'test_datatype')


def test_remove_component_with_artifact(db):
    # Test artifact is deleted from artifact store
    component_with_artifact = TestComponent(
        identifier='test_with_artifact',
        version=0,
        artifact={'test': 'test'},
    )
    db.apply(component_with_artifact)
    info_with_artifact = db.metadata.get_component(
        'test-component', 'test_with_artifact', 0
    )
    artifact_file_id = info_with_artifact['artifact'][1:]
    with patch.object(db.artifact_store, '_delete_bytes') as mock_delete:
        db._remove_component_version(
            'test-component', 'test_with_artifact', 0, force=True
        )
        mock_delete.assert_called_once_with(artifact_file_id)


def test_remove_one_version(db):
    for component in [
        TestComponent(identifier='test', version=0),
        TestComponent(identifier='test', version=1),
    ]:
        db.apply(component)

    # Only remove the version
    db.remove('test-component', 'test', 1, force=True)
    # Wait for the db to update
    time.sleep(0.1)
    assert db.show('test-component', 'test') == [0]


def test_remove_multi_version(db):
    for component in [
        TestComponent(identifier='test', version=0),
        TestComponent(identifier='test', version=1),
        TestComponent(identifier='test', version=2),
    ]:
        db.apply(component)

    db.remove('test-component', 'test', force=True)
    # Wait for the db to update
    time.sleep(0.1)
    assert db.show('test-component', 'test') == []


def test_remove_not_exist_component(db):
    with pytest.raises(FileNotFoundError) as e:
        db.remove('test-component', 'test', 0, force=True)
    assert 'test' in str(e)

    db.remove('test-component', 'test', force=True)


def test_show(db):
    for component in [
        TestComponent(identifier='a1'),
        TestComponent(identifier='a2'),
        TestComponent(identifier='a3'),
        TestComponent(identifier='b'),
        TestComponent(identifier='b'),
        TestComponent(identifier='b'),
        DataType(identifier='c1'),
        DataType(identifier='c2'),
    ]:
        db.apply(component)

    with pytest.raises(ValueError) as e:
        db.show('test-component', version=1)
    assert 'None' in str(e) and '1' in str(e)

    assert sorted(db.show('test-component')) == ['a1', 'a2', 'a3', 'b']
    assert sorted(db.show('datatype')) == ['c1', 'c2']

    assert sorted(db.show('test-component', 'a1')) == [0]
    assert sorted(db.show('test-component', 'b')) == [0, 1, 2]

    # Test get specific version
    info = db.show('test-component', 'b', 1)
    assert isinstance(info, dict)
    assert info['version'] == 1
    assert info['identifier'] == 'b'
    assert info['_path'].split('.')[-1] == 'TestComponent'

    # Test get last version
    assert db.show('test-component', 'b', -1)['version'] == 2


def test_load(db):
    m1 = ObjectModel(object=lambda x: x, identifier='m1', datatype='int32')

    components = [
        DataType(identifier='e1'),
        DataType(identifier='e2'),
        m1,
        ObjectModel(object=lambda x: x, identifier='m1', datatype='int32'),
        m1,
    ]
    for component in components:
        db.apply(component)

    # Test load fails
    # error version
    with pytest.raises(Exception):
        db.load('datatype', 'e1', version=1)

    # # error identifier
    with pytest.raises(Exception):
        db.load('model', 'e1')

    datatype = db.load('datatype', 'e1')
    assert isinstance(datatype, DataType)

    assert datatype.type_id, datatype.identifier in db.cluster.cache


def test_insert(db):
    db.cfg.auto_schema = True
    add_fake_model(db)
    inserted_ids = (
        db['documents'].insert([{'x': i, 'update': True} for i in range(5)]).execute()
    )
    assert len(inserted_ids) == 5

    uuid = db.show('listener', 'listener-x', 0)['uuid']

    key = f'_outputs__listener-x__{uuid}'
    new_docs = db[key].select().execute()
    result = [doc[key] for doc in new_docs]
    assert sorted(result) == ['0', '1', '2', '3', '4']


def test_insert_artifacts(db):
    dt = DataType(
        'my_saveable',
        encodable='artifact',
        encoder=pickle_encode,
        decoder=pickle_decode,
    )
    table = Table(
        'documents',
        schema=Schema('documents', fields={'x': dt}),
    )
    db.apply(table)
    db._insert(
        db['documents'].insert(
            [Document({'x': numpy.random.randn(100)}) for _ in range(1)]
        )
    )
    r = list(db.execute(db['documents'].select()))[0]
    assert isinstance(r['x'], numpy.ndarray)


def test_insert_sql_db(db):
    db.cfg.auto_schema = True
    listener = add_fake_model(db)
    table = db['documents']
    inserted_ids = db.execute(
        table.insert([Document({'id': str(i), 'x': i}) for i in range(5)])
    )
    assert len(inserted_ids) == 5

    q = table.select().outputs(listener.predict_id)

    new_docs = db.execute(q)
    new_docs = list(new_docs)

    result = [Document(doc.unpack())[listener.outputs] for doc in new_docs]
    assert sorted(result) == ['0', '1', '2', '3', '4']


@pytest.mark.skipif(not mongodb_config, reason='MongoDB not configured')
def test_update_db(db):
    # TODO: test update sql db after the update method is implemented
    add_fake_model(db)
    q = db['documents'].insert([Document({'x': i, 'update': True}) for i in range(5)])
    db._insert(q)
    updated_ids, _ = db._update(
        db['documents'].update_many({}, Document({'$set': {'x': 100}}))
    )
    assert len(updated_ids) == 5
    listener_uuid = db.show('listener')[0].split('/')[-1]
    key = f'_outputs__{listener_uuid}'
    new_docs = db[key].select().execute()
    for doc in new_docs:
        assert doc[key]
        doc = Document(doc.unpack())

        # TODO: Need to support Update result in predict_in_db
        # assert doc[key] == '100'


def test_replace(db):
    model = ObjectModel(
        object=lambda x: x + 1,
        identifier='m',
        datatype=DataType(identifier='base'),
    )
    model.version = 0
    with pytest.raises(Exception):
        db.replace(model)

    db.replace(model, upsert=True)

    assert db.load('model', 'm').predict(1) == 2

    new_model = ObjectModel(
        object=lambda x: x + 2, identifier='m', signature='singleton'
    )
    new_model.version = 0
    db.replace(new_model)
    time.sleep(0.1)
    assert db.load('model', 'm').predict_batches([1]) == [3]

    # replace the last version of the model
    new_model = ObjectModel(
        object=lambda x: x + 3, identifier='m', signature='singleton'
    )
    db.replace(new_model)
    time.sleep(0.1)
    assert db.load('model', 'm').predict_batches([1]) == [4]


def test_replace_with_child(db):
    db.apply(Table('docs', schema=Schema('docs', fields={'X': 'int', 'y': 'int'})))
    trainer = Trainer(
        identifier='trainer',
        key=('X', 'y'),
        select=db['docs'].select(),
    )
    model = FakeModel(
        identifier='m123',
        trainer=trainer,
    )

    with patch.object(FakeModel, 'fit'):
        db.apply(model)

    model_ids = db.show('model')

    reloaded = db.load('model', model.identifier)

    assert 'm123' in model_ids
    assert hasattr(reloaded, 'trainer')

    assert isinstance(reloaded.trainer, Trainer)
    assert not reloaded.metric_values

    model.trainer.metric_values['acc'] = [1, 2, 3]
    db.replace(model)

    rereloaded = db.load('model', model.identifier)
    assert isinstance(rereloaded.trainer, Trainer)
    assert rereloaded.trainer.metric_values

    db.remove('table', 'docs', force=True)

    db.metadata.get_component_version_parents(rereloaded.uuid)


def test_compound_component(db):
    m = ObjectModel(
        object=lambda x: x + 1,
        identifier='my-test-module',
        datatype=FieldType(identifier='int'),
    )

    db.apply(m)
    assert 'my-test-module' in db.show('model')
    assert db.show('model', 'my-test-module') == [0]

    db.apply(m)
    assert db.show('model', 'my-test-module') == [0]

    db.apply(
        ObjectModel(
            object=lambda x: x + 1,
            identifier='my-test-module',
            datatype=FieldType(identifier='int'),
        )
    )
    assert db.show('model', 'my-test-module') == [0, 1]

    m = db.load(type_id='model', identifier='my-test-module')
    assert isinstance(m.datatype, FieldType)

    db.remove('model', 'my-test-module', force=True)


def test_reload_dataset(db):
    from test.utils.setup.fake_data import add_random_data

    from superduper.components.dataset import Dataset

    add_random_data(db, n=6)

    select = db['documents'].select().filter(db['documents']['_fold'] == 'valid')

    d = Dataset(
        identifier='my_valid',
        select=select,
        sample_size=100,
    )
    db.apply(d)
    new_d = db.load('dataset', 'my_valid')
    assert new_d.sample_size == 100


def test_dataset(db):
    db.cfg.auto_schema = True
    from test.utils.setup.fake_data import add_random_data

    add_random_data(db, n=6)
    select = db['documents'].select().filter(db['documents']['_fold'] == 'valid')

    d = Dataset(
        identifier='test_dataset',
        select=select,
    )
    db.apply(d)
    assert db.show('dataset') == ['test_dataset']
    dataset = db.load('dataset', 'test_dataset')
    assert len(dataset.data) == len(list(db.execute(dataset.select)))


def test_delete_componet_with_same_artifact(db):
    from superduper import ObjectModel

    model1 = ObjectModel(
        object=lambda x: x + 1,
        identifier='model1',
    )

    model2 = ObjectModel(
        object=model1.object,
        identifier='model2',
    )

    db.apply(model1)
    db.apply(model2)

    db.remove('model', 'model1', force=True)
    model2 = db.load('model', 'model2')
    model2.init()
    assert model2.predict(1) == 2


def test_retry_on_token_expiry(db):
    # Mock the methods
    db.retry = 1

    def check_retry():
        if db.retry == 1:
            db.retry = 0
            raise Exception("The connection token has been expired already")
        else:
            return True

    db.databackend._backend.test_retry = check_retry

    with patch.object(db.databackend._backend, 'reconnect') as reconnect:
        with patch.object(
            db.databackend._backend,
            'test_retry',
            side_effect=check_retry,
        ) as mock_test_retry:
            db.databackend.test_retry()
            assert reconnect.call_count == 1
            assert mock_test_retry.call_count == 2
