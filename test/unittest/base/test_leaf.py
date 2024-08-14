import dataclasses as dc
import typing as t
from pprint import pprint

from superduper import ObjectModel
from superduper.backends.base.query import Query
from superduper.base.constant import KEY_BUILDS
from superduper.base.document import Document
from superduper.base.leaf import Leaf
from superduper.components.component import Component


class Test(Leaf):
    b: t.Optional[str] = 'a'
    c: t.Optional[t.Union[float, str]] = 1.0
    a: t.Optional[int] = 1


class OtherSer(Leaf):
    d: str = 'd'


class TestSubModel(Component):
    type_id: t.ClassVar[str] = 'test-sub-model'
    a: int = 1
    b: str = 'b'
    c: ObjectModel = dc.field(
        default_factory=ObjectModel(identifier='test-2', object=lambda x: x + 2)
    )
    d: t.List[ObjectModel] = dc.field(default_factory=[])
    e: OtherSer = dc.field(default_factory=OtherSer(identifier='test', d='test'))
    f: t.Callable = dc.field(default=lambda x: x)


class MySer(Leaf):
    a: int = 1
    b: str = 'b'
    c: Leaf = dc.field(default_factory=OtherSer(identifier='test', d='test'))


def test_encode_leaf():
    obj = Test('test', a=1, b='test_b', c=1.5)
    assert obj.dict().encode() == {
        '_path': 'test.unittest.base.test_leaf.Test',
        'uuid': obj.uuid,
        'identifier': 'test',
        'a': 1,
        'b': 'test_b',
        'c': 1.5,
        '_builds': {},
        '_files': {},
        '_blobs': {},
    }


def test_encode_leaf_with_children():
    obj = MySer(
        identifier='my_ser',
        a=1,
        b='test_b',
        c=OtherSer(identifier='other_ser', d='test'),
    )
    assert obj.dict().encode() == {
        '_path': 'test.unittest.base.test_leaf.MySer',
        'identifier': 'my_ser',
        'uuid': obj.uuid,
        'a': 1,
        'b': 'test_b',
        'c': '?other_ser',
        '_builds': {
            'other_ser': {
                k: v for k, v in obj.c.dict().unpack().items() if k != 'identifier'
            },
        },
        '_files': {},
        '_blobs': {},
    }


def test_save_variables_2():
    query = (
        Query(table='documents')
        .like({'x': '<var:X>'}, vector_index='test')
        .find({'x': {'$regex': '^test/1'}})
    )

    assert [x for x in query.variables] == ['X']


def test_saveable():
    s = MySer(identifier='sr', a=1, b='test', c=OtherSer(identifier='other', d='test'))
    r = Document(s.dict()).encode()
    print(r)


def test_component_with_document():
    t = TestSubModel(
        identifier='test-1',
        a=2,
        b='test',
        c=ObjectModel('test-2', object=lambda x: x + 2),
        d=[ObjectModel('test-3', object=lambda x: x + 2)],
        e=OtherSer(identifier='other', d='test'),
        f=lambda x: x,
    )
    print('encoding')
    d = Document(t.dict())
    r = d.encode()
    builds = r[KEY_BUILDS]

    pprint(r)
    assert len(builds) == 8

    for leaf in builds:
        print(type(leaf))


def test_find_variables():
    from superduper import Document

    r = Document({'txt': '<var:test>'})

    assert r.variables == ['test']

    q = Query(table='test').find_one(Document({'txt': '<var:test>'}))

    assert q.variables == ['test']

    q = (
        Query(table='test')
        .like(Document({'txt': '<var:test>'}), vector_index='test')
        .find()
        .limit(5)
    )

    q_set = q.set_variables(test='my-value')

    assert q_set.variables == []
