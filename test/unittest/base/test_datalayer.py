import os
import time
import typing as t  # noqa: F401
from typing import Any, ClassVar, Optional, Sequence
from unittest.mock import patch

import numpy
import pytest

from superduper.base.datalayer import Datalayer
from superduper.base.datatype import (
    BaseDataType,
    Blob,
    dill_serializer,
)
from superduper.components.component import Component
from superduper.components.dataset import Dataset
from superduper.components.listener import Listener
from superduper.components.model import Model, ObjectModel, Trainer
from superduper.components.table import Table
from superduper.misc import typing as st


class FakeModel(Model):
    """Fake model for testing."""

    ...


n_data_points = 250

ibis_config = os.environ.get('SUPERDUPER_CONFIG', "").endswith('ibis.yaml')
mongodb_config = os.environ.get('SUPERDUPER_CONFIG', "").endswith('mongodb.yaml')


class TestComponent(Component):
    breaks: ClassVar[Sequence] = ('inc',)
    _fields = {'artifact': dill_serializer}
    inc: int = 0
    is_on_create: bool = False
    is_after_create: bool = False
    check_clean_up: bool = False
    child: Optional[Component] = None
    artifact: Any = None

    def on_create(self):
        self.is_after_create = True

    def cleanup(self):
        if self.check_clean_up:
            raise Exception('cleanup')

    @property
    def artifact_attributes(self):
        return ['artifact'] if self.artifact else []

    @property
    def child_components(self):
        return [('child', 'TestComponent')] if self.child else []


def add_fake_model(db: Datalayer):

    table = Table(identifier='documents', fields={'x': 'int', 'id': 'str'})

    db.apply(table)

    model = ObjectModel(
        object=lambda x: str(x),
        identifier='fake_model',
    )

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
    # assert component.is_after_create is True
    assert component.version == 0
    assert db.show('TestComponent', 'test') == [0]

    # Test the component saves the data correctly
    component_loaded = db.load('TestComponent', 'test')

    original_serialized = component.dict().encode()
    saved_serialized = component_loaded.dict().encode()

    assert original_serialized['identifier'] == saved_serialized['identifier']

    # Check duplicate components are not added
    db.apply(component)

    component_loaded = db.load('TestComponent', 'test')

    assert component_loaded.version == 0
    assert db.show('TestComponent', 'test') == [0]

    # Check the version is incremented
    component = TestComponent(identifier='test', inc=1)
    db.apply(component)
    assert component.version == 1
    assert db.show('TestComponent', 'test') == [0, 1]

    component = TestComponent(identifier='test', inc=2)
    db.apply(component)
    assert component.version == 2
    assert db.show('TestComponent', 'test') == [0, 1, 2]


class TestComponentPickle(TestComponent):
    artifact: st.Pickle


def test_add_component_with_bad_artifact(db):
    artifact = {'data': lambda x: x}
    component = TestComponentPickle(
        identifier='test',
        artifact=artifact,
    )
    with pytest.raises(Exception):
        db.apply(component)


def test_add_artifact_auto_replace(db):
    # Check artifact is automatically replaced to metadata
    artifact = {'data': 1}
    component = TestComponent(identifier='test', artifact=artifact)
    db.apply(component)
    r = db.show('TestComponent', 'test', -1)
    assert r['artifact'].startswith('&')


def test_add_child(db: Datalayer):
    child_component = TestComponent(identifier='child')
    component = TestComponent(identifier='test', child=child_component)

    db.apply(component)
    assert db.show('TestComponent', 'test') == [0]
    assert db.show('TestComponent', 'child') == [0]

    parents = db.metadata.get_component_version_parents(child_component.uuid)
    assert parents == [(component.component, component.uuid)]

    child_component_2 = TestComponent(identifier='child-2')
    component_3 = TestComponent(identifier='test-3', child=child_component_2)
    db.apply(component_3)
    assert db.show('TestComponent', 'test-3') == [0]
    assert db.show('TestComponent', 'child-2') == [0]

    parents = db.metadata.get_component_version_parents(child_component_2.uuid)
    assert parents == [(component_3.component, component_3.uuid)]


def test_add(db):
    component = TestComponent(identifier='test')
    db.apply(component)
    assert db.show('TestComponent', 'test') == [0]

    for component in [
        TestComponent(identifier='test_list_1'),
        TestComponent(identifier='test_list_2'),
    ]:
        db.apply(component)
    assert db.show('TestComponent', 'test_list_1') == [0]
    assert db.show('TestComponent', 'test_list_2') == [0]

    with pytest.raises(ValueError):
        db.apply('test')


def test_add_with_artifact(db):
    m = ObjectModel(
        identifier='test',
        object=lambda x: x + 2,
        datatype='artifact',
    )

    db.apply(m)

    m = db.load('ObjectModel', m.identifier)

    assert m.object is not None

    assert isinstance(m.object, Blob)
    m.setup()
    assert callable(m.object)


def test_add_table(db):
    component = Table('test', fields={'field': 'str'})
    db.apply(component)


def test_remove_multi_version(db):
    for component in [
        TestComponent(identifier='test', inc=0),
        TestComponent(identifier='test', inc=1),
        TestComponent(identifier='test', inc=2),
    ]:
        db.apply(component)

    db.remove('TestComponent', 'test', force=True)
    # Wait for the db to update
    time.sleep(0.1)
    assert db.show('TestComponent', 'test') == []


def test_show(db):
    for component in [
        TestComponent(identifier='a1'),
        TestComponent(identifier='a2'),
        TestComponent(identifier='a3'),
        TestComponent(identifier='b', inc=0),
        TestComponent(identifier='b', inc=1),
        TestComponent(identifier='b', inc=2),
        # DataType(identifier='c1'),
        # DataType(identifier='c2'),
    ]:
        db.apply(component)

    with pytest.raises(ValueError) as e:
        db.show('TestComponent', version=1)
    assert 'None' in str(e) and '1' in str(e)

    assert sorted(db.show('TestComponent')) == ['a1', 'a2', 'a3', 'b']
    # assert sorted(db.show('datatype')) == ['c1', 'c2']

    assert sorted(db.show('TestComponent', 'a1')) == [0]
    assert sorted(db.show('TestComponent', 'b')) == [0, 1, 2]

    # Test get specific version
    info = db.show('TestComponent', 'b', 1)
    assert isinstance(info, dict)
    assert info['version'] == 1
    assert info['identifier'] == 'b'
    assert info['_path'].split('.')[-1] == 'TestComponent'

    # Test get last version
    assert db.show('TestComponent', 'b', -1)['version'] == 2


class DataType(BaseDataType):
    def encode_data(self, item):
        return item

    def decode_data(self, item):
        return item


def test_load(db):
    m1 = ObjectModel(object=lambda x: x, identifier='m1')

    components = [
        # DataType(identifier='e1'),
        # DataType(identifier='e2'),
        m1,
        ObjectModel(object=lambda x: x, identifier='m1'),
    ]
    for component in components:
        db.apply(component)

    # # Test load fails
    # # error version
    # with pytest.raises(Exception):
    #     db.load('datatype', 'e1', version=1)

    # # error identifier
    with pytest.raises(Exception):
        db.load('ObjectModel', 'e1')

    # datatype = db.load('datatype', 'e1')
    # assert isinstance(datatype, BaseDataType)

    # assert datatype.type_id, datatype.identifier in db.cluster.cache


def test_insert(db):
    add_fake_model(db)
    inserted_ids = db['documents'].insert([{'x': i} for i in range(5)])
    assert len(inserted_ids) == 5

    uuid = db.show('Listener', 'listener-x', 0)['uuid']

    key = f'_outputs__listener-x__{uuid}'
    new_docs = db[key].select().execute()
    result = [doc[key] for doc in new_docs]
    assert sorted(result) == ['0', '1', '2', '3', '4']


def test_insert_artifacts(db):

    db.apply(Table('documents', fields={'x': 'array[float:100]'}))

    db['documents'].insert([{'x': numpy.random.randn(100)} for _ in range(1)])
    r = db['documents'].get()
    assert isinstance(r['x'], numpy.ndarray)


def my_lambda(x):
    return x + 1


def test_compound_component(db):
    m = ObjectModel(object=my_lambda, identifier='my-test-module', datatype='int')

    db.apply(m)
    assert 'my-test-module' in db.show('ObjectModel')
    assert db.show('ObjectModel', 'my-test-module') == [0]

    db.apply(
        ObjectModel(object=lambda x: x + 2, identifier='my-test-module', datatype='int')
    )
    assert db.show('ObjectModel', 'my-test-module') == [0, 1]

    m = db.load('ObjectModel', identifier='my-test-module')

    db.remove('ObjectModel', 'my-test-module', force=True)


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
    new_d = db.load('Dataset', 'my_valid')
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
    assert db.show('Dataset') == ['test_dataset']
    dataset = db.load('Dataset', 'test_dataset')
    assert len(dataset.data) == len(dataset.select.execute())


def test_delete_component_with_same_artifact(db):
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

    db.remove('ObjectModel', 'model1', force=True)

    model2 = db.load('ObjectModel', 'model2')
    model2.setup()
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


class Container(Component):
    model: Model


def test_remove_not_recursive(db: Datalayer):

    m = Container(
        identifier='container',
        model=FakeModel('test'),
    )

    db.apply(m)

    assert db.show('Container', 'container') == [0]

    db.remove('Container', 'container')

    assert 'container' not in db.show('Container')

    assert 'test' in db.show('FakeModel')


def test_remove_recursive(db: Datalayer):

    m = Container(
        identifier='container',
        model=FakeModel('test'),
    )

    db.apply(m)

    assert db.show('Container', 'container') == [0]

    db.remove('Container', 'container', recursive=True)

    assert 'container' not in db.show('Container')

    assert 'test' not in db.show('FakeModel')


def test_remove_shared(db: Datalayer):

    fm = FakeModel('test')

    m1 = Container(identifier='container1', model=fm)
    m2 = Container(identifier='container2', model=fm)

    db.apply(m1)
    db.apply(m2)

    assert db.show('Container', 'container1') == [0]
    assert db.show('Container', 'container2') == [0]

    with pytest.raises(Exception) as e:
        db.remove('Container', 'container1', recursive=True)

    assert 'container2' in str(e)

    # Check that the container is not removed, since a child failed
    assert 'container1' in db.show('Container')

    db.remove('Container', 'container2', recursive=True, force=True)

    assert 'container2' not in db.show('Container')
    assert 'container1' in db.show('Container')
    assert 'test' in db.show('FakeModel')
