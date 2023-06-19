import typing as t

from superduperdb.misc.logger import logging

if t.TYPE_CHECKING:
    from superduperdb.datalayer.base.database import BaseDatabase


class BasePlaceholder:
    """
    Base placeholders, used to signify an attribute was saved separately.
    """


class DBPlaceholder(BasePlaceholder):
    """
    Placeholder for a database connection
    """

    is_database = True


class Placeholder(BasePlaceholder):
    """
    Placeholder.
    """

    def __init__(self, identifier: str, variety: str, version: t.Optional[int] = None):
        self.identifier = identifier
        self.variety = variety
        self.version = version


class PlaceholderList(BasePlaceholder, list):
    """
    List of placeholders.
    """

    def __init__(self, variety: str, *args, **kwargs):
        super().__init__(
            [Placeholder(arg, variety) for arg in args[0]], *args[1:], **kwargs
        )
        self.variety = variety

    def __repr__(self):
        return f'PlaceholderList[{self.variety}](' + super().__repr__() + ')'

    def aslist(self) -> t.List[str]:
        return [x.identifier for x in self]


class BaseComponent:
    """
    Essentially just there to put Component and ComponentList on common ground.
    """

    def _set_subcomponent(self, key, value):
        logging.warn(f'Setting {value} component at {key}')
        super().__setattr__(key, value)

    def __setattr__(self, key, value):
        try:
            current = getattr(self, key)
            # don't allow surgery on component, since messes with version rules
            if isinstance(current, BaseComponent) or isinstance(current, Placeholder):
                raise Exception('Cannot set component attribute!')
        except AttributeError:
            pass
        return super().__setattr__(key, value)


class Component(BaseComponent):
    """
    Base component which models, watchers, learning tasks etc. inherit from.

    :param identifier: Unique ID
    """

    variety: str
    repopulate_on_init = False

    def __init__(self, identifier: str):
        self.identifier: str = identifier
        self.version: t.Optional[int] = None

    @property
    def unique_id(self):
        if self.version is None:
            raise Exception('Version not yet set for component uniqueness')
        return f'{self.variety}/{self.identifier}/{self.version}'

    @property
    def child_components(self) -> t.List['Component']:
        out = []
        for v in vars(self).values():
            if isinstance(v, Component):
                out.append(v)
            elif isinstance(v, ComponentList):
                out.extend(list(v))
        return out

    @property
    def child_references(self) -> t.List[Placeholder]:
        out = []
        for v in vars(self).values():
            if isinstance(v, Placeholder):
                out.append(v)
            elif isinstance(v, PlaceholderList):
                out.extend(list(v))
        return out

    def repopulate(self, database: 'BaseDatabase'):  # noqa: F821 why?
        """
        Set all attributes which were separately saved and serialized.

        :param database: Database connector responsible for saving/ loading components
        """

        def reload(object):
            if isinstance(object, Placeholder):
                reloaded = database.load(
                    variety=object.variety,
                    identifier=object.identifier,
                    version=object.version,
                    allow_hidden=True,
                )
                return reload(reloaded)

            if isinstance(object, PlaceholderList):
                reloaded = [
                    database.load(c.variety, c.identifier, allow_hidden=True)
                    for c in object
                ]
                for i, c in enumerate(reloaded):
                    reloaded[i] = c.repopulate(database)
                return ComponentList(object.variety, reloaded)

            if isinstance(object, DBPlaceholder):
                return database

            return object

        items = [
            (k, v) for k, v in vars(self).items() if isinstance(v, BasePlaceholder)
        ]
        has_a_db = False
        for k, v in items:
            reloaded = reload(v)
            self.__dict__[k] = reloaded
            if getattr(v, 'is_database', False):
                has_a_db = True

        if has_a_db and hasattr(self, '_post_attach_database'):
            self._post_attach_database()

        return self

    def asdict(self):
        return {'identifier': self.identifier}

    def was_stripped(self) -> bool:
        """
        Test if all contained BaseComponent attributes were stripped
        (no longer part of object)
        """
        return not any(isinstance(v, BaseComponent) for v in vars(self).values())

    def __repr__(self):
        super_repr = super().__repr__()
        parts = super_repr.split(' object at ')
        subcomponents = [
            getattr(self, attr)
            for attr in vars(self)
            if isinstance(getattr(self, attr), BaseComponent)
            or isinstance(getattr(self, attr), BasePlaceholder)
        ]
        if not subcomponents:
            return super_repr
        lines = [str(subcomponent) for subcomponent in subcomponents]
        lines = [parts[0], *['    ' + x for x in lines], parts[1]]
        return '\n'.join(lines)

    def schedule_jobs(self, database):
        return []

    @classmethod
    def make_unique_id(cls, variety, identifier, version):
        return f'{variety}/{identifier}/{version}'


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
                self[i] = database.load(self.variety, item)
            self[i], _ = self[i].repopulate(database)

    def aslist(self):
        return [c.identifier for c in self]


def strip(component: BaseComponent):
    """
    Strip component down to object which doesn't contain a BaseComponent part.
    This may be applied so that objects aren't redundantly serialized and replaced
    in multiple places.

    :param component: component to be stripped
    """
    from superduperdb.datalayer.base.database import BaseDatabase

    assert isinstance(component, BaseComponent)
    if isinstance(component, Placeholder):
        return component
    if isinstance(component, ComponentList):
        return PlaceholderList(component.variety, [strip(obj) for obj in component])
    for attr in vars(component):
        subcomponent = getattr(component, attr)
        if isinstance(subcomponent, ComponentList):
            component._set_subcomponent(attr, strip(subcomponent))
        elif isinstance(subcomponent, Component):
            component._set_subcomponent(
                attr,
                Placeholder(
                    subcomponent.identifier,
                    subcomponent.variety,
                    subcomponent.version,
                ),
            )
        elif isinstance(subcomponent, BaseDatabase):
            component._set_subcomponent(attr, None)
    return component


def is_placeholders_or_components(items: t.Union[t.List[t.Any], t.Tuple]):
    """
    Test whether the list is just strings and also test whether it's just components
    """

    is_placeholders = all([isinstance(y, str) for y in items])
    is_components = all([isinstance(y, Component) for y in items])
    return is_placeholders, is_components
