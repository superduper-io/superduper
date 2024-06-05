import typing as t

from .component import Component


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
