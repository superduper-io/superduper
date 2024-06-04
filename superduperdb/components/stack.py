import dataclasses as dc
import typing as t

from superduperdb.misc.annotations import merge_docstrings

from .component import Component


@merge_docstrings
class Stack(Component):
    """
    A placeholder to hold list of components under a namespace.

    :param components: List of components to stack together and add to database.
    """

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
