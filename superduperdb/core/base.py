
class BasePlaceholder:
    """
    Base placeholders, used to signify an attribute was saved separately.
    """


class Placeholder(BasePlaceholder):
    """
    Placeholder.
    """
    def __init__(self, identifier: str):
        self.identifier = identifier


class PlaceholderList(BasePlaceholder, list):
    """
    List of placeholders.
    """
    def __repr__(self):
        return 'PlaceholderList(' + super().__repr__() + ')'


class BaseComponent:
    """
    Base component which models, watchers, learning tasks etc. inherit from.
    """


class Component(BaseComponent):

    def __init__(self, identifier: str):
        self.identifier = identifier

    def repopulate(self, database: 'superduperdb.datalayer.base.BaseDatabase'):
        """
        Set all attributes which were separately saved and serialized.

        :param database: Database connector which is reponsible for saving/ loading components
        """
        for attr in dir(self):
            object = getattr(self, attr)
            if not isinstance(object, BasePlaceholder):
                continue
            if isinstance(reloaded, Placeholder):
                reloaded = database.load_component(object.identifier)
                reloaded = reloaded.repopulate(database)
            elif isinstance(reloaded, PlaceholderList):
                reloaded = [database.load_component(c.identifier) for c in object]
                for i, c in enumerate(reloaded):
                    reloaded[i] = database.repopulate(database)
                reloaded = ComponentList(reloaded)
            setattr(self, attr, reloaded)
        return self

    def was_stripped(self) -> bool:
        """
        Test if all contained BaseComponent attributes were stripped (no longer part of object).
        """
        for attr in dir(self):
            if isinstance(getattr(self, attr), BaseComponent):
                return False
        return True

    def __repr__(self):
        super_repr = super().__repr__()
        parts = super_repr.split(' object at ')
        subcomponents = [
            getattr(self, attr) for attr in dir(self)
            if isinstance(getattr(self, attr), BaseComponent)
            or isinstance(getattr(self, attr), BasePlaceholder)
        ]
        if not subcomponents:
            return super_repr
        lines = [str(subcomponent) for subcomponent in subcomponents]
        lines = [parts[0], *['    '  + x for x in lines], parts[1]]
        return '\n'.join(lines)


class ComponentList(BaseComponent, list):
    """
    List of base components.
    """
    def __repr__(self):
        return 'ComponentList(' + super().__repr__() + ')'


def strip(component: BaseComponent):
    """
    Strip component down to object which doesn't contain a BaseComponent part.

    :param component: component to be stripped
    """
    assert isinstance(component, BaseComponent)
    if isinstance(component, Placeholder):
        return component
    if isinstance(component, ComponentList):
        return PlaceholderList([strip(obj) for obj in component])
    for attr in dir(component):
        subcomponent = getattr(component, attr)
        if isinstance(subcomponent, ComponentList):
            setattr(component, attr, strip(subcomponent))
        elif isinstance(subcomponent, BaseComponent):
            setattr(component, attr, Placeholder(subcomponent.identifier))
    return component

