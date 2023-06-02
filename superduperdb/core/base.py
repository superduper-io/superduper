from superduperdb.misc.logger import logging


class BasePlaceholder:
    """
    Base placeholders, used to signify an attribute was saved separately.
    """


class Placeholder(BasePlaceholder):
    """
    Placeholder.
    """
    def __init__(self, identifier: str, variety: str):
        self.identifier = identifier
        self.variety = variety


class PlaceholderList(BasePlaceholder, list):
    """
    List of placeholders.
    """
    def __init__(self, variety, *args, **kwargs):
        super().__init__([Placeholder(arg, variety) for arg in args[0]], *args[1:], **kwargs)
        self.variety = variety

    def __repr__(self):
        return f'PlaceholderList[{self.variety}](' + super().__repr__() + ')'

    def aslist(self):
        return [x.identifier for x in self]


class BaseComponent:
    """
    Essentially just there to put Component and ComponentList on common ground.
    """


class Component(BaseComponent):
    """
    Base component which models, watchers, learning tasks etc. inherit from.

    :param identifier: Unique ID
    """

    def __init__(self, identifier: str):
        self.identifier = identifier

    def repopulate(self, database: 'superduperdb.datalayer.base.BaseDatabase'):
        """
        Set all attributes which were separately saved and serialized.

        :param database: Database connector which is reponsible for saving/ loading components
        """
        def reload(object):
            if isinstance(object, Placeholder):
                reloaded = database.load_component(object.identifier, variety=object.variety)
                return reloaded.repopulate(database)

            if isinstance(object, PlaceholderList):
                reloaded = [database.load_component(c.identifier, c.variety) for c in object]
                for i, c in enumerate(reloaded):
                    reloaded[i] = c.repopulate(database)
                return ComponentList(object.variety, reloaded)

            return object

        items = ((k, v) for k, v in vars(self).items() if isinstance(v, BasePlaceholder))
        self.__dict__.update((k, reload(v)) for k, v in items)
        return self

    def asdict(self):
        return {'identifier': self.identifier}

    def was_stripped(self) -> bool:
        """
        Test if all contained BaseComponent attributes were stripped (no longer part of object).
        """
        return not any(isinstance(v, BaseComponent) for v in vars(self).values())

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

    def schedule_jobs(self, database):
        return []


class ComponentList(BaseComponent, list):
    """
    List of base components.
    """
    def __init__(self, variety, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variety = variety

    def __repr__(self):
        return f'ComponentList[{self.variety}](' + super().__repr__() + ')'

    def repopulate(self, database):
        for i, item in enumerate(self):
            if isinstance(item, str):
                self[i] = database.load_component(item)
            self[i] = self[i].repopulate(database)

    def aslist(self):
        return [c.identifier for c in self]


def strip(component: BaseComponent):
    """
    Strip component down to object which doesn't contain a BaseComponent part.
    This may be applied so that objects aren't redundantly serialized and replaced in multiple
    places.

    :param component: component to be stripped
    """
    assert isinstance(component, BaseComponent)
    if isinstance(component, Placeholder):
        return component
    if isinstance(component, ComponentList):
        return PlaceholderList(component.variety, [strip(obj) for obj in component])
    for attr in dir(component):
        subcomponent = getattr(component, attr)
        if isinstance(subcomponent, ComponentList):
            setattr(component, attr, strip(subcomponent))
        elif isinstance(subcomponent, BaseComponent):
            setattr(component, attr, Placeholder(subcomponent.identifier, subcomponent.variety))
    return component


def is_placeholders_or_components(items: list):
    """
    Test whether the list is just strings and also test whether it's just components
    """
    is_placeholders = all([isinstance(y, str) for y in items])
    is_components = all([isinstance(y, Component) for y in items])
    return is_placeholders, is_components
