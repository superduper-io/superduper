import dataclasses as dc
import hashlib
import os
import shutil
import tempfile
import typing as t

import pytest

from superduper import ObjectModel, Table
from superduper.base.annotations import trigger
from superduper.base.datatype import (
    Blob,
    _Artifact,
    dill_serializer,
)
from superduper.base.metadata import JOB_PHASE_FAILED, Job
from superduper.components.component import Component
from superduper.components.listener import Listener


@pytest.fixture
def cleanup():
    yield
    try:
        os.remove("test_export.tar.gz")
        shutil.rmtree("test_export")
    except FileNotFoundError:
        pass


@dc.dataclass(kw_only=True)
class MyComponent(Component):
    type_id: t.ClassVar[str] = "my_type"
    _fields = {
        'my_dict': dill_serializer,
        'nested_list': dill_serializer,
    }
    my_dict: t.Dict
    nested_list: t.List
    a: t.Callable


def test_reload(db):
    m = ObjectModel('test', object=lambda x: x + 1)

    db.apply(m)

    reloaded = db.load('ObjectModel', 'test')
    reloaded.setup()

    assert reloaded.object(1) == 2


def test_init(db, monkeypatch):
    a = Blob(
        identifier="456",
        bytes=dill_serializer._encode_data(lambda x: x + 1),
        db=db,
        builder=dill_serializer._decode_data,
    )
    my_dict = Blob(
        identifier="456",
        bytes=dill_serializer._encode_data({'a': lambda x: x + 1}),
        db=db,
        builder=dill_serializer._decode_data,
    )

    list_ = Blob(
        identifier='789',
        bytes=dill_serializer._encode_data([lambda x: x + 1]),
        db=db,
        builder=dill_serializer._decode_data,
    )

    c = MyComponent("test", my_dict=my_dict, a=a, nested_list=list_)

    c.setup()

    assert callable(c.my_dict["a"])
    assert c.my_dict["a"](1) == 2

    assert callable(c.a)
    assert c.a(1) == 2

    assert callable(c.nested_list[0])
    assert c.nested_list[0](1) == 2


def test_load_lazily(db):
    m = ObjectModel("lazy_model", object=lambda x: x + 2)

    db.apply(m)

    reloaded = db.load("ObjectModel", m.identifier)

    assert isinstance(reloaded.object, Blob)
    assert reloaded.object.bytes is None

    reloaded.setup()

    assert callable(reloaded.object)


def test_export_and_read():
    m = ObjectModel("test", object=lambda x: x + 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "tmp_save")
        m.export(save_path)
        assert os.path.exists(os.path.join(tmpdir, "tmp_save", "blobs"))

        def load(blob):
            with open(blob, "rb") as f:
                return f.read()

        reloaded = Component.read(save_path)  # getters=getters

        assert isinstance(reloaded, ObjectModel)


def test_set_variables(db):
    m = Listener(
        identifier="test",
        model=ObjectModel(
            identifier="test",
            object=lambda x: x + 2,
        ),
        key="<var:key>",
        select=db["docs"],
    )

    listener = m.set_variables(key="key_value", docs="docs_value")
    assert listener.key == "key_value"


def test_encoding(db):
    m = Listener(
        identifier="test",
        model=ObjectModel(
            identifier="object_model",
            object=lambda x: x + 2,
        ),
        key="X",
        select=db["docs"].select(),
    )

    r = m.encode()

    assert isinstance(r['model'], str)


class UpstreamComponent(Component):
    @trigger('apply')
    def a_job(self):
        with open(f'upstream_{self.uuid}.txt', 'w'):
            pass


class MyListener(Listener):
    @trigger('apply')
    def my_trigger(self):
        uuid = self.upstream[0].uuid
        assert os.path.exists(f'upstream_{uuid}.txt')
        return []


@pytest.fixture
def clean():
    yield
    os.system('rm upstream_*.txt')


def test_upstream(db, clean):
    from superduper import Table

    db.apply(Table('docs', fields={'id': 'str', 'x': 'str'}))
    c1 = UpstreamComponent(identifier='c1')
    m = MyListener(
        identifier='l1',
        upstream=[c1],
        model=ObjectModel(
            identifier="model1",
            object=lambda x: x + 2,
        ),
        key="x",
        select=db["docs"].select(),
    )

    db.apply(m)


class NewComponent(Component): ...


@pytest.mark.skip("Awaiting refactor of version by version deletion")
def test_remove_recursive(db):
    c1 = NewComponent(identifier='c1')
    c2 = NewComponent(identifier='c2', upstream=[c1])
    c3 = NewComponent(identifier='c3', upstream=[c2, c1])

    db.apply(c3)

    assert sorted([r['identifier'] for r in db.show()]) == [
        'NewComponent',
        'c1',
        'c2',
        'c3',
    ]

    db.remove('NewComponent', c3.identifier, recursive=True, force=True)

    len(db.show()) == 1


class MyClass:
    def __init__(self, a):
        self.a = a

    def __hash__(self):
        h = hashlib.sha256(str(self.__dict__).encode()).hexdigest()
        return int(h, 16)

    def predict(self, x):
        import numpy

        return numpy.random.randn(20)


def test_calls_post_init():
    t = Table('test', fields={'x': 'str'})
    assert hasattr(t, 'version')


def my_func(x):
    return x + 1


def test_rehash(db):

    # as a fallback, users can bring a custom hash function

    m1 = ObjectModel(
        identifier='model',
        object=MyClass(1),
    )

    m2 = ObjectModel(
        identifier='model',
        object=MyClass(1),
    )

    assert m1.hash == m2.hash

    m3 = ObjectModel(
        identifier='model',
        object=MyClass(2),
    )

    assert m1.hash != m3.hash

    m4 = ObjectModel(
        identifier='model',
        object=my_func,
    )

    m5 = ObjectModel(
        identifier='model',
        object=my_func,
    )

    assert m4.hash == m5.hash

    m6 = ObjectModel(
        identifier='model',
        object=my_func,
    )

    db.apply(m6)

    reloaded = db.load('ObjectModel', 'model')

    db.apply(
        Listener(
            'test',
            model=m6,
            select=db['documents'],
            upstream=[Table('documents', fields={'x': 'str'})],
            key='x',
        )
    )

    assert m6.hash == reloaded.hash


def test_propagate_failure(db):

    m = ObjectModel(
        identifier='model',
        object=my_func,
    )

    listener = Listener(
        model=m,
        key='x',
        identifier='test',
        select=db['documents'],
        upstream=[Table('documents', fields={'x': 'str'})],
    )

    db.apply(listener, force=True)

    try:
        raise Exception("Test exception")
    except Exception as e:
        m.propagate_failure(e)

    status_model = db.load('ObjectModel', 'model').status
    status_listener = db.load('Listener', 'test').status

    assert status_model['phase'] == JOB_PHASE_FAILED
    assert status_listener['phase'] == JOB_PHASE_FAILED
    import datetime

    job = Job(
        context='123',
        component='ObjectModel',
        identifier='model',
        uuid=m.uuid,
        args=(2,),
        kwargs={},
        time=str(datetime.datetime.now()),
        job_id='my-job-id',
        method='predict',
    )

    db.metadata.create_job(job.dict())

    db.metadata.set_job_status(
        job_id='my-job-id',
        status_update={'phase': JOB_PHASE_FAILED, 'reason': 'Test exception'},
    )

    j = db['Job'].get(job_id='my-job-id')

    assert j['status']['phase'] == JOB_PHASE_FAILED
    assert j['status']['reason'] == 'Test exception'

    list = db.load('Listener', 'test')

    # check that a message has propagated to the listener
    assert 'Job' in str(list.status['children'])
