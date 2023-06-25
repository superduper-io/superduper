from superduperdb.core.base import Component, ComponentList, strip, restore


class ComponentA(Component):
    variety = 'a'


class ComponentB(Component):
    variety = 'b'


class MyComponent(Component):
    def __init__(self, identifier, components: list):
        super().__init__(identifier)
        self.things = ComponentList('many', components)
        self.whatever = ComponentA('other')


def test_strip():
    comp = MyComponent('my_component', [ComponentA('a'), ComponentB('b')])
    assert not comp.was_stripped()
    stripped_comp, cache = strip(comp)
    print(stripped_comp)
    assert stripped_comp.was_stripped()
    comp = restore(stripped_comp, cache)
    print(comp)
