import dataclasses as dc
import typing as t
from pprint import pprint

from superduperdb.backends.mongodb.query import Collection
from superduperdb.base.serializable import Serializable, Variable


@dc.dataclass
class Test(Serializable):
    a: int
    b: t.Union[str, Variable]
    c: t.Union[float, Variable]


def test_serializable_variables_1():
    r = Test(a=1, b='test/1', c=1.5)
    assert r.serialize() == {
        'cls': 'Test',
        'dict': {'a': 1, 'b': 'test/1', 'c': 1.5},
        'module': 'test.unittest.base.test_serializable',
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

    assert s.set_variables(db=Tmp(version=1)).serialize() == r.serialize()


def test_serializable_variables_2():
    query = (
        Collection(Variable('Collection'))
        .like({'x': Variable('X')}, vector_index='test')
        .find({'x': {'$regex': '^test/1'}})
    )

    assert [x.value for x in query.variables] == ['Collection', 'X']

    q = Collection(Variable('Collection')).find({'x': Variable('X')})
    print(pprint(q.serialize()))
