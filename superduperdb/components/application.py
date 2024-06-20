import typing as t

from superduperdb.base.datalayer import Datalayer

from .component import Component


class Application(Component):
    """
    A placeholder to hold list of components with associated funcionality.

    :param components: List of components to group together and apply to `superduperdb`.
    :param namespace: List of tuples with type_id and identifier of components to
        assist in managing application.
    """

    literals: t.ClassVar[t.Sequence[str]] = ('template',)
    type_id: t.ClassVar[str] = 'application'
    components: t.Sequence[Component]
    namespace: t.Optional[t.Sequence[t.Tuple[str, str]]] = None

    def pre_create(self, db: Datalayer):
        """Pre-create hook.

        :param db: Datalayer instance
        """
        self.namespace = [
            {'type_id': c.type_id, 'identifier': c.identifier} for c in self.children
        ]
        return super().pre_create(db)

    def cleanup(self, db: Datalayer):
        """Cleanup hook.

        :param db: Datalayer instance
        """
        if self.namespace is not None:
            for type_id, identifier in self.namespace:
                db.remove(type_id=type_id, identifier=identifier, force=True)
        return super().cleanup(db)
