from superduperdb.core.base import Component, ComponentList, strip


class ComponentA(Component):
    ...


class ComponentB(Component):
    ...


class MyComponent(Component):
    def __init__(self, identifier, components: list):
        super().__init__(identifier)
        self.things = ComponentList(components)
        self.whatever = ComponentA('other')


def test_strip():
    comp = MyComponent('my_component', [ComponentA('a'), ComponentB('b')])
    assert not comp.was_stripped()
    stripped_comp = strip(comp)
    print(stripped_comp)
    assert stripped_comp.was_stripped()
