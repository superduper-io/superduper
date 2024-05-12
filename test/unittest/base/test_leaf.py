import dataclasses as dc
import typing as t
from pprint import pprint

from superduperdb import ObjectModel
from superduperdb.backends.mongodb.query import MongoQuery
from superduperdb.base.document import Document
from superduperdb.base.leaf import Leaf
from superduperdb.base.variables import Variable
from superduperdb.components.component import Component


@dc.dataclass
class Test(Leaf):
    b: t.Optional[t.Union[str, Variable]] = 'a'
    c: t.Optional[t.Union[float, Variable]] = 1.0
    a: t.Optional[int] = 1


@dc.dataclass
class OtherSer(Leaf):
    d: str = 'd'


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
class MySer(Leaf):
    a: int
    b: str
    c: Leaf


def test_serialize_variables_1():
    r = Test(a=1, b='test/1', c=1.5)
    assert r.dict().encode() == {
        '_path': 'test.unittest.base.test_leaf.Test',
        'a': 1,
        'b': 'test/1',
        'c': 1.5,
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


def test_save_variables_2():
    query = (
        MongoQuery(Variable('Collection'))
        .like({'x': Variable('X')}, vector_index='test')
        .find({'x': {'$regex': '^test/1'}})
    )

    assert [x.value for x in query.variables] == ['Collection', 'X']

    q = MongoQuery(Variable('Collection')).find({'x': Variable('X')})
    print(pprint(q.dict()))


def test_saveable():
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


def test_find_variables():
    from superduperdb import Document
    from superduperdb.backends.mongodb import MongoQuery
    from superduperdb.base.variables import Variable

    r = Document({'txt': Variable('test')})

    assert [str(x) for x in r.variables] == ['$test']

    q = MongoQuery('test').find_one(Document({'txt': Variable('test')}))

    assert [str(x) for x in q.variables] == ['$test']

    q = (
        MongoQuery('test')
        .like(Document({'txt': Variable('test')}), vector_index='test')
        .find()
        .limit(5)
    )

    q_set = q.set_variables(None, test='my-value')

    assert q_set.variables == []
