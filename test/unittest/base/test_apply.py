import typing as t

from superduper import Component
from superduper.base.annotations import trigger
from superduper.base.apply import _apply
from superduper.base.datalayer import Datalayer
from superduper.components.application import Application


class MyValidator(Component):
    target: int

    def run(self, parent):
        return float(parent.b * parent.c > self.target)


class MyComponent(Component):
    type_id: t.ClassVar[str] = 'my'
    breaks: t.ClassVar[t.Sequence[str]] = ('b',)

    a: str
    b: int
    c: float = 0.5
    sub: Component | None = None
    validate_results: bool | None = None

    @trigger('apply', requires='sub')
    def validate_in_db(self):
        if isinstance(self.sub, MyValidator):
            self.validate_results = self.sub.run(self)
            self.db.replace(self)
        return self.validate_results


def test_simple_apply(db: Datalayer):
    c = MyComponent('test', a='value', b=1)

    db.apply(c)

    assert db.show('my') == ['test']
    assert db.show('my', 'test') == [0]


def test_skip_same(db: Datalayer):
    c = MyComponent('test', a='value', b=1)

    db.apply(c)

    assert db.show()

    c = MyComponent('test', a='value', b=1)

    db.apply(c)

    # applying same component again,
    # means nothing happens
    # no computations and no updates
    assert db.show('my', 'test') == [0]


def test_update_component_version(db: Datalayer):
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


def test_break_version(db: Datalayer):
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


def test_update_nested(db: Datalayer):
    c = MyComponent(
        'test',
        a='value',
        b=2,
        sub=MyComponent(
            'sub',
            a='sub-value',
            b=3,
        ),
    )

    db.apply(c)

    assert set(db.show('my')) == {'test', 'sub'}

    c = MyComponent(
        'test',
        a='value',
        b=2,
        sub=MyComponent(
            'sub',
            a='new-sub-value',
            b=3,
        ),
    )

    # Neither the child or the parent
    # is broken by updating sub.a
    # that means we don't get a new
    # version
    db.apply(c)

    assert db.show('my', 'test') == [0]
    assert db.show('my', 'sub') == [0]

    # Nonetheless the child is updated
    assert db.load('my', 'sub').a == 'new-sub-value'


def test_break_nested(db: Datalayer):
    c = MyComponent(
        'test',
        a='value',
        b=2,
        sub=MyComponent(
            'sub',
            a='sub-value',
            b=3,
        ),
    )

    db.apply(c)

    assert set(db.show('my')) == {'test', 'sub'}

    c = MyComponent(
        'test',
        a='value',
        b=2,
        sub=MyComponent(
            'sub',
            a='sub-value',
            b=4,
        ),
    )

    # re-applying this component should break
    # the child, however the parent isn't
    # broken by self.sub, so is only updated
    db.apply(c)

    assert db.show('my', 'test') == [0]
    assert db.show('my', 'sub') == [0, 1]
    assert db.load('my', 'sub').b == 4


def test_job_on_update(db: Datalayer):
    c = MyComponent(
        'test',
        a='value',
        b=2,
    )

    db.apply(c)

    assert db.show('my', 'test') == [0]

    c = MyComponent('test', a='value', b=2, sub=MyValidator('valid', target=2))

    db.apply(c)

    reload = db.load('my', 'test')

    assert reload.validate_results is not None


def test_duplicate_job_submission(db: Datalayer):
    from superduper import ObjectModel

    db.cfg.auto_schema = True

    def my_func(x):
        return x + 1

    listener_1 = ObjectModel(
        'list1',
        object=my_func,
    ).to_listener(key='x', select=db['docs'].select())

    listener_2 = ObjectModel(
        'list2',
        object=my_func,
        upstream=listener_1,
    ).to_listener(key=listener_1.outputs, select=db[listener_1.outputs].select())

    c = Application(
        'test',
        components=[
            listener_1,
            listener_2,
        ],
    )

    db['docs'].insert([{'x': i} for i in range(10)]).execute()

    db.apply(c)


def test_diff(db):
    c = MyComponent(
        'test',
        a='value',
        b=2,
        sub=MyComponent(
            'sub',
            a='sub-value',
            b=3,
        ),
    )

    db.apply(c)

    assert set(db.show('my')) == {'test', 'sub'}

    c = MyComponent(
        'test',
        a='value',
        b=2,
        sub=MyComponent(
            'sub',
            a='sub-value',
            b=4,
        ),
    )

    # re-applying this component should break
    # the child, however the parent isn't
    # broken by self.sub, so is only updated
    diff = {}
    _apply(
        db=db,
        object=c,
        global_diff=diff,
    )

    import json

    print(json.dumps(diff, indent=2))

    assert set(diff['sub']['changes'].keys()) == {'b'}
    assert diff['sub']['changes']['b'] == 4

    db.apply(c)
