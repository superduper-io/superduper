import typing as t

from .component import Component


class Application(Component):
    """
    A placeholder to hold list of components with associated 
    funcionality under a namespace.

    :param components: List of components to group together and apply to `superduperdb`.
    """

    type_id: t.ClassVar[str] = 'application'
    components: t.Sequence[Component]