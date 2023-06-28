import typing as t
from contextlib import contextmanager

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

    def __init__(
        self,
        identifier: str,
        variety: str,
        version: t.Optional[int] = None,
        id: t.Optional[int] = None,
    ):
        self.identifier = identifier
        self.variety = variety
        self.version = version
        self.id = id


class PlaceholderList(BasePlaceholder):
    """
    List of placeholders.
    """

    # ruff: noqa: E501
    def __init__(
        self,
        variety: str,
        placeholders: t.Union[t.List[str], t.List[Placeholder]],  # TODO - fix this type
    ):
        self.placeholders = placeholders
        if placeholders and isinstance(self.placeholders[0], str):
            self.placeholders = [Placeholder(arg, variety) for arg in placeholders]  # type: ignore[arg-type]
        elif placeholders:
            assert isinstance(self.placeholders[0], Placeholder)
            self.placeholders = placeholders
        else:
            self.placeholders = placeholders
        self.variety = variety

    def __iter__(self):
        return iter(self.placeholders)

    def __getitem__(self, item):
        return self.placeholders[item]

    def aslist(self) -> t.List[str]:
        return [x.identifier for x in self.placeholders]  # type: ignore[union-attr]


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
                out.extend(v.placeholders)  # type: ignore[arg-type]
        return out

    @contextmanager
    def saving(self):
        try:
            print('Stripping sub-components to references')
            cache = strip(self, top_level=True)[1]
            yield self
        finally:
            restore(self, cache)

    def repopulate(self, database: 'BaseDatabase'):  # noqa: F821 why?
        """
        Set all attributes which were separately saved and serialized.

        :param database: Database connector responsible for saving/ loading components
        """

        def reload(object: t.Any) -> t.Any:
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

    def schedule_jobs(self, database: 'BaseDatabase') -> t.List:
        return []

    @classmethod
    def make_unique_id(cls, variety, identifier, version):
        return f'{variety}/{identifier}/{version}'


class ComponentList(BaseComponent):
    """
    List of base components.
    """

    def __init__(self, variety, components):
        self.variety = variety
        self.components = components

    def __getitem__(self, item):
        return self.components[item]

    def __iter__(self):
        return iter(self.components)

    def repopulate(self, database):
        for i, item in enumerate(self):
            if isinstance(item, str):
                self[i] = database.load(self.variety, item)
            self[i], _ = self[i].repopulate(database)

    def aslist(self):
        return [c.identifier for c in self]


def strip(component: BaseComponent, top_level=True) -> t.Union[BaseComponent, 'BaseDatabase', BasePlaceholder]:
    """
    Strip component down to object which doesn't contain a BaseComponent part.
    This may be applied so that objects aren't redundantly serialized and replaced
    in multiple places.

    :param component: component to be stripped
    """
    from superduperdb.datalayer.base.database import BaseDatabase

    cache = {}
    assert isinstance(component, BaseComponent)
    if isinstance(component, ComponentList):
        placeholders = []
        for sc in component:
            stripped, subcache = strip(sc, top_level=False)
            cache.update(subcache)
            placeholders.append(stripped)
        return PlaceholderList(component.variety, placeholders), cache
    for attr in vars(component):
        subcomponent = getattr(component, attr)
        if isinstance(subcomponent, ComponentList):
            sub_component, sub_cache = strip(subcomponent, top_level=False)
            component._set_subcomponent(attr, sub_component)
            cache.update(sub_cache)
        elif isinstance(subcomponent, Component):
            cache[id(subcomponent)] = subcomponent
            component._set_subcomponent(
                attr,
                Placeholder(
                    subcomponent.identifier,
                    subcomponent.variety,
                    subcomponent.version,
                    id=id(subcomponent),
                ),
            )
        elif isinstance(subcomponent, BaseDatabase):
            cache[id(subcomponent)] = subcomponent
            component._set_subcomponent(attr, None)
    if top_level:
        return component, cache
    else:
        cache[id(component)] = component
        return (
            Placeholder(
                component.identifier,  # type: ignore
                component.variety,  # type: ignore
                component.version,  # type: ignore
                id=id(component),
            ),
            cache,
        )


# ruff: noqa: E501
def restore(component: t.Union[BaseComponent, BasePlaceholder], cache: t.Dict):
    if isinstance(component, PlaceholderList):
        return ComponentList(component.variety, [restore(c, cache) for c in component.placeholders])  # type: ignore[arg-type]
    if isinstance(component, Placeholder):
        try:
            return cache[component.id]
        except KeyError:
            logging.warn(f'Left placeholder {component} because not found in cache')
            return component
    for attr, value in vars(component).items():
        if isinstance(value, BasePlaceholder):
            component._set_subcomponent(attr, restore(value, cache))  # type: ignore[union-attr]
    return component


def is_placeholders_or_components(items: t.Union[t.List[t.Any], t.Tuple]) -> t.Tuple[bool, bool]:
    """
    Test whether the list is just strings and also test whether it's just components
    """

    is_placeholders = all([isinstance(y, str) for y in items])
    is_components = all([isinstance(y, Component) for y in items])
    return is_placeholders, is_components
