import dataclasses as dc
import typing as t
from pprint import pprint

from superduperdb import ObjectModel
from superduperdb.backends.mongodb.query import Collection
from superduperdb.base.document import Document
from superduperdb.base.serializable import Serializable, Variable
from superduperdb.components.component import Component


@dc.dataclass
class Test(Serializable):
    a: int
    b: t.Union[str, Variable]
    c: t.Union[float, Variable]


@dc.dataclass
class OtherSer(Serializable):
    d: str


@dc.dataclass(kw_only=True)
class TestSubModel(Component):
    type_id: t.ClassVar[str] = 'test-sub-model'
    a: int
    b: t.Union[str, Variable]
    c: ObjectModel
    d: t.List[ObjectModel]
    e: OtherSer
    f: t.Callable


@dc.dataclass
class MySer(Serializable):
    a: int
    b: str
    c: Serializable


def test_serializable_variables_1():
    r = Test(a=1, b='test/1', c=1.5)
    assert r.encode() == {
        '_content': {
            'cls': 'Test',
            'dict': {'a': 1, 'b': 'test/1', 'c': 1.5},
            'leaf_type': 'serializable',
            'module': 'test.unittest.base.test_serializable',
        }
    }
    s = Test(
        a=1,
        b=Variable(
            'test/{version}', lambda db, value, kwargs: value.format(version=db.version)
        ),
        c=Variable('number', lambda db, value, kwargs: db[value]),
    )

    @dc.dataclass
    class Tmp:
        version: int

        def __getitem__(self, item):
            return {'number': 1.5}[item]

    assert s.set_variables(db=Tmp(version=1)).encode() == r.encode()


def test_serializable_variables_2():
    query = (
        Collection(Variable('Collection'))
        .like({'x': Variable('X')}, vector_index='test')
        .find({'x': {'$regex': '^test/1'}})
    )

    assert [x.value for x in query.variables] == ['Collection', 'X']

    q = Collection(Variable('Collection')).find({'x': Variable('X')})
    print(pprint(q.serialize()))


def test_serializable():
    s = MySer(a=1, b='test', c=OtherSer(d='test'))
    r = Document(s.dict()).encode()
    print(r)


def test_component_with_document():
    t = TestSubModel(
        identifier='test-1',
        a=2,
        b='test',
        c=ObjectModel('test-2', object=lambda x: x + 2),
        d=[ObjectModel('test-3', object=lambda x: x + 2)],
        e=OtherSer(d='test'),
        f=lambda x: x,
    )
    print('encoding')
    d = Document(t.dict())
    r = d.encode()
    leaves = d.get_leaves()

    import pprint

    pprint.pprint(r)

    print(r)

    print(leaves)
    assert len(leaves) == 4

    for leaf in leaves:
        print(type(leaf))


def test_compound_select_serialize():
    q = Collection('test').find({}).limit(5)

    r = q.dict().encode()

    s = Serializable.decode(r)

    print(s)
