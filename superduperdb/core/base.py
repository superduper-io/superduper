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
        return 'PlaceholderList(' + super().__repr__() + ')'

    def aslist(self):
        return [x.identifier for x in self]


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
            try:
                object = getattr(self, attr)
            except Exception as e:
                logging.warn(str(e))
                continue
            if not isinstance(object, BasePlaceholder):
                continue
            if isinstance(object, Placeholder):
                reloaded = database.load_component(object.identifier, variety=object.variety)
                reloaded = reloaded.repopulate(database)
            elif isinstance(object, PlaceholderList):
                reloaded = [database.load_component(c.identifier, c.variety) for c in object]
                for i, c in enumerate(reloaded):
                    reloaded[i] = c.repopulate(database)
                reloaded = ComponentList(object.variety, reloaded)
            setattr(self, attr, reloaded)
        return self

    def asdict(self):
        return {'identifier': self.identifier}

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
        return 'ComponentList(' + super().__repr__() + ')'

    def repopulate(self, database):
        for i, item in enumerate(self):
            if isinstance(item, str):
                self[i] = database.load_component(item)
            self[i] = self[i].repopulate(database)

    def aslist(self):
        return [c.identifier for i, c in enumerate(self)]


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

