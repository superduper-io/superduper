import dataclasses as dc
import typing as t

from superduperdb.base.serializable import Serializable, Variable


@dc.dataclass
class Test(Serializable):
    a: int
    b: t.Union[str, Variable]
    c: t.Union[float, Variable]


def test_serializable():
    r = Test(a=1, b='test/1', c=1.5)
    assert r.serialize() == {
        'cls': 'Test',
        'dict': {'a': 1, 'b': 'test/1', 'c': 1.5},
        'module': 'test.unittest.base.test_serializable',
    }
    s = Test(
        a=1,
        b=Variable(
            'test/{version}', lambda db, value: value.format(version=db.version)
        ),
        c=Variable('number', lambda db, value: db[value]),
    )

    @dc.dataclass
    class Tmp:
        version: int

        def __getitem__(self, item):
            return {'number': 1.5}[item]

    assert s.set_variables(db=Tmp(version=1)).serialize() == r.serialize()
