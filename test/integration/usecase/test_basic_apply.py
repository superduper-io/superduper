from superduper import Component


class TestApply(Component):
    a: str
    b: Component | None = None


def test(db):

    component = TestApply('test', a="hello", b=TestApply('test_b', a="world"))

    r = component.dict()

    assert 'version' in r

    db.apply(component)

    reloaded = db.load('TestApply', 'test')
    reloaded.setup()

    reloaded.show()
