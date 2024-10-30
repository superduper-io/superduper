import pytest
import typing as t
from superduper import Component

class MyComponent(Component):
    type_id: t.ClassVar[str] = 'my'
    breaks: t.ClassVar[t.Sequence[str]] = ('b',)

    a: str
    b: int
    sub: Component | None = None


def test_simple_apply(db):

    c = MyComponent('test', a='value', b=1)

    db.apply(c)

    assert db.show('my') == ['test']


@pytest.mark.skip
def test_skip_same(db):

    c = MyComponent('test', a='value', b=1)

    db.apply(c)

    c = MyComponent('test', a='value', b=1)

    db.apply(c)

    # applying same component again,
    # means nothing happens
    # no computations and no updates
    assert db.show('my', 'test') == [0]


@pytest.mark.skip
def test_update_component_version(db):

    c = MyComponent('test', a='value', b=1)

    db.apply(c)

    reload = db.load('my', 'test')

    assert reload.a == 'value'

    c = MyComponent('test', a='new-value', b=1)

    db.apply(c)

    # creates only one version but
    # updates it
    assert db.show('my', 'test') == [0]

    reload = db.load('my', 'test')

    assert reload.a == 'new-value'


def test_break_version(db):

    c = MyComponent('test', a='value', b=1)

    db.apply(c)

    reload = db.load('my', 'test')

    assert reload.b == 1

    c = MyComponent('test', a='value', b=2)

    db.apply(c)

    # creates only one version but
    # updates it
    assert db.show('my', 'test') == [0, 1]

    reload = db.load('my', 'test')

    assert reload.b == 2


@pytest.mark.skip
def test_update_nested(db):

    c = MyComponent(
        'test',
        a='value',
        b=2,
        sub=MyComponent(
            'sub',
            a='sub-value',
            b=3,
        )
    )

    db.apply(c)

    assert db.show('my') == ['test', 'sub']

    c = MyComponent(
        'test',
        a='value',
        b=2,
        sub=MyComponent(
            'sub',
            a='new-sub-value',
            b=3,
        )
    )

    db.apply(c)

    assert db.show('my', 'test') == [0]
    assert db.show('my', 'sub') == [0]


@pytest.mark.skip
def test_break_nested(db):

    c = MyComponent(
        'test',
        a='value',
        b=2,
        sub=MyComponent(
            'sub',
            a='sub-value',
            b=3,
        )
    )

    db.apply(c)

    assert db.show('my') == ['test', 'sub']

    c = MyComponent(
        'test',
        a='value',
        b=2,
        sub=MyComponent(
            'sub',
            a='sub-value',
            b=4,
        )
    )

    # re-applying this component should break 
    # the child, however the parent isn't
    # broken by self.sub, so is only updated
    db.apply(c)

    assert db.show('my', 'test') == [0]
    assert db.show('my', 'sub') == [0, 1]