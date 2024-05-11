import dataclasses as dc
import typing as t

from superduperdb.base.document import _build_leaves
from superduperdb.misc.annotations import public_api

from .component import Component

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer


@public_api(stability='alpha')
@dc.dataclass(kw_only=True)
class Stack(Component):
    """Component to hold a list of components under a namespace and package.

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
        """Datalayer property."""
        return self._db

    @db.setter
    def db(self, value):
        """Datalayer setter.

        :param value: Item to set the property.
        """
        self._db = value
        for component in self.components:
            component.db = value

    @staticmethod
    def from_list(identifier, content, db: t.Optional['Datalayer'] = None):
        """Helper method to create a Stack from a list `content`.

        :param identifier: Unique identifier.
        :param content: Content to create a stack.
        :param db: Datalayer instance.
        """
        out, exit = _build_leaves(content, db=db)
        out = [out[k] for k in exit]
        return Stack(identifier, components=out)
