from superduperdb.server.registry import Registry
from argparse import Namespace
import typing as t
import superduperdb as s


def setup_registry(register):
    class One(s.JSONable):
        one = 'one'

    class Two(One):
        two = 'two'

    class Three(One):
        two = 'three'

    class Object:
        @register
        def first(self, one: One) -> Two:
            return Two(**one.dict())

        @register
        def second(self, one: One, three: Three) -> One:
            return one

    return Namespace(**locals())


def test_registry():
    registry = Registry()
    test = setup_registry(registry.register)
    assert registry.Parameter == t.Union[test.One, test.Three]
    assert registry.Result == t.Union[test.One, test.Two]

    return registry
