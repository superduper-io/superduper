import dataclasses as dc
import typing as t

from superduperdb.misc.annotations import public_api

from .component import Component

if t.TYPE_CHECKING:
    pass


@public_api(stability='alpha')
@dc.dataclass(kw_only=True)
class Stack(Component):
    """
    A placeholder to hold a list of components under a namespace and
    package them as a tarball.
    This tarball can be retrieved back to a Stack instance with the
    ``load`` method.

    {component_parameters}
    :param components: List of components to stack together and add to
                       the database.
    """

    __doc__ = __doc__.format(component_parameters=Component.__doc__)

    type_id: t.ClassVar[str] = 'stack'
    components: t.Sequence[Component]

    @property
    def db(self):
        """
        Datalayer property.
        """
        return self._db

    @db.setter
    def db(self, value):
        """
        Datalayer setter.

        :param value: Item to set the property.
        """
        self._db = value
        for component in self.components:
            component.db = value
