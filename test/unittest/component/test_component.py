import dataclasses as dc
import os
import shutil
import tempfile
import typing as t

import pytest

from superduper import ObjectModel, Schema, Table
from superduper.base.annotations import trigger
from superduper.components.component import Component
from superduper.components.datatype import (
    BaseDataType,
    Blob,
    dill_serializer,
)
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

    reloaded = db.load('model', 'test')
    reloaded.unpack()


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

    c.init()

    assert callable(c.my_dict["a"])
    assert c.my_dict["a"](1) == 2

    assert callable(c.a)
    assert c.a(1) == 2

    assert callable(c.nested_list[0])
    assert c.nested_list[0](1) == 2


def test_load_lazily(db):
    m = ObjectModel("lazy_model", object=lambda x: x + 2)

    db.apply(m)
    db.expire(m.uuid)

    reloaded = db.load("model", m.identifier)

    assert isinstance(reloaded.object, Blob)
    assert reloaded.object.bytes is None

    reloaded.init()

    assert callable(reloaded.object)


def test_export_and_read():
    m = ObjectModel("test", object=lambda x: x + 2, datatype=dill_serializer)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "tmp_save")
        m.export(save_path)
        assert os.path.exists(os.path.join(tmpdir, "tmp_save", "blobs"))

        def load(blob):
            with open(blob, "rb") as f:
                return f.read()

        reloaded = Component.read(save_path)  # getters=getters

        assert isinstance(reloaded, ObjectModel)
        assert isinstance(reloaded.datatype, BaseDataType)


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

    from superduper import Document

    e = m.encode()
    recon = Document.decode(e).unpack()

    recon.init()

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
    from superduper import Schema, Table

    db.apply(Table('docs', schema=Schema('docs/schema', fields={'id': str, 'x': str})))
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


class NewComponent(Component):
    ...


def test_remove_recursive(db):
    c1 = NewComponent(identifier='c1')
    c2 = NewComponent(identifier='c2', upstream=[c1])
    c3 = NewComponent(identifier='c3', upstream=[c2, c1])

    db.apply(c3)

    assert sorted([r['identifier'] for r in db.show()]) == ['c1', 'c2', 'c3']

    db.remove('component', c3.identifier, recursive=True, force=True)

    assert not db.show()


class MyClass:
    def predict(self, x):
        import numpy

        return numpy.random.randn(20)


def test_calls_post_init():
    t = Table('test', schema=Schema('test', fields={'x': 'str'}))
    assert hasattr(t, 'version')
