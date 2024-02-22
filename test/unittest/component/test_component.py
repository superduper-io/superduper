import dataclasses as dc
import os
import shutil
import typing as t
from test.db_config import DBConfig

import pytest

from superduperdb import ObjectModel
from superduperdb.base.document import Document
from superduperdb.components.component import Component
from superduperdb.components.datatype import (
    Artifact,
    Empty,
    LazyArtifact,
    dill_serializer,
)


@pytest.fixture
def cleanup():
    yield
    try:
        os.remove('test_export.tar.gz')
        shutil.rmtree('test_export')
    except FileNotFoundError:
        pass


def test_compile_decompile(cleanup):
    m = ObjectModel('test_export', object=lambda x: x, datatype=dill_serializer)
    m.version = 0
    m.datatype.version = 0
    m.export()
    assert os.path.exists('test_export.tar.gz')
    m_reload = Component.import_('test_export.tar.gz')
    assert isinstance(m_reload, ObjectModel)


@dc.dataclass(kw_only=True)
class MyComponent(Component):
    type_id: t.ClassVar[str] = 'my_type'
    _lazy_fields: t.ClassVar[t.Sequence[str]] = ('my_dict',)
    my_dict: t.Dict
    nested_list: t.List
    a: t.Callable


def test_init(monkeypatch):
    from unittest.mock import MagicMock

    def unpack(self, db):
        if '_base' in self.keys():
            return [lambda x: x + 1, lambda x: x + 2]
        return {'a': lambda x: x + 1}

    monkeypatch.setattr(Document, 'unpack', unpack)

    e = Artifact(x=None, file_id='123', datatype=dill_serializer)
    a = Artifact(x=None, file_id='456', datatype=dill_serializer)

    def side_effect(*args, **kwargs):
        a.x = lambda x: x + 1

    a.init = MagicMock()
    a.init.side_effect = side_effect

    list_ = [e, a]

    c = MyComponent('test', my_dict={'a': a}, a=a, nested_list=list_)

    c.init()

    assert callable(c.my_dict['a'])
    assert c.my_dict['a'](1) == 2

    assert callable(c.a)
    assert c.a(1) == 2

    assert callable(c.nested_list[1])
    assert c.nested_list[1](1) == 3


@pytest.mark.parametrize("db", [DBConfig.mongodb], indirect=True)
def test_load_lazily(db):
    m = ObjectModel('lazy_model', object=lambda x: x + 2)

    db.add(m)

    reloaded = db.load('model', m.identifier)

    assert isinstance(reloaded.object, LazyArtifact)
    assert isinstance(reloaded.object.x, Empty)

    reloaded.init()

    assert callable(reloaded.object)
