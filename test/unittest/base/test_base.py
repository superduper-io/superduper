import dataclasses as dc
import json
import typing as t
from pprint import pprint

from superduper import CFG, ObjectModel
from superduper.base.base import Base
from superduper.base.constant import KEY_BLOBS, KEY_BUILDS
from superduper.base.document import Document
from superduper.components.component import Component
from superduper.components.listener import Listener


class Test(Base):
    a: t.Optional[int] = 1
    b: t.Optional[str] = 'a'
    c: t.Optional[float] = 1.0


class OtherSer(Base):
    d: str = 'd'


class TestSubModel(Component):
    _fields = {'c': 'component', 'd': 'slist', 'e': 'leaf', 'f': 'default'}

    type_id: t.ClassVar[str] = 'test-sub-model'
    a: int = 1
    b: str = 'b'
    c: ObjectModel | None = None
    d: t.List[ObjectModel] = dc.field(default_factory=[])
    e: OtherSer | None = None
    f: t.Callable


class MySer(Base):
    _fields = {'c': 'leaf'}

    a: int = 1
    b: str = 'b'
    c: Base = dc.field(default_factory=OtherSer(d='test'))


def test_insert_and_recall(db):
    data = [Test(a=i, b='test_b', c=1.5) for i in range(10)]
    db.insert(data)
    r = db['Test'].get()
    assert r is not None


def test_encode_leaf():
    obj = Test(a=1, b='test_b', c=1.5)
    assert obj.dict().encode(keep_schema=False) == {
        '_path': 'test.unittest.base.test_base.Test',
        'a': 1,
        'b': 'test_b',
        'c': 1.5,
        '_builds': {},
        '_files': {},
        '_blobs': {},
    }


def test_encode_leaf_with_children():
    obj = MySer(
        a=1,
        b='test_b',
        c=OtherSer(d='test'),
    )
    assert obj.dict().encode(keep_schema=False) == {
        '_path': 'test.unittest.base.test_base.MySer',
        'a': 1,
        'b': 'test_b',
        'c': (
            obj.c.dict().unpack()
            if CFG.json_native
            else json.dumps(obj.c.dict().unpack())
        ),
        '_builds': {},
        '_files': {},
        '_blobs': {},
    }


def test_save_variables_2(db):
    t = db['documents']
    query = t.like({'x': '<var:X>'}, vector_index='test').filter(t['x'] == 1)

    assert [x for x in query.variables] == ['X']


def test_save_non_string_variables(db):
    query = db['documents'].select().limit('<var:limit>')
    assert str(query) == 'documents.select().limit("<var:limit>")'
    assert [x for x in query.variables] == ['limit']


def test_saveable():
    s = MySer(a=1, b='test', c=OtherSer(d='test'))
    r = Document(s.dict()).encode()
    print(r)


def test_component_with_document():
    t = TestSubModel(
        identifier='test-1',
        a=2,
        b='test',
        c=ObjectModel(identifier='test-2', object=lambda x: x + 2),
        d=[ObjectModel(identifier='test-3', object=lambda x: x + 2)],
        e=OtherSer(d='test'),
        f=lambda x: x,
    )
    print('encoding')
    d = t.dict()
    r = d.encode(leaves_to_keep=Base)
    builds = r[KEY_BUILDS]

    pprint(r)
    assert len(builds) == 2
    assert len(r[KEY_BLOBS]) == 1

    for leaf in builds:
        print(type(builds[leaf]))


def test_find_variables(db):
    from superduper import Document

    r = Document({'txt': '<var:test>'})

    assert r.variables == ['test']

    t = db['test']

    q = t.filter(t['txt'] == '<var:test>')

    assert q.variables == ['test']

    q = db['test'].like({'txt': '<var:test>'}, vector_index='test').limit(5)

    q_set = q.set_variables(test='my-value')

    assert q_set.variables == []
