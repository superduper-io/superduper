from superduperdb import Component, superduper

from . import command


@command(help='Apply the stack tarball to the database')
def apply(path: str):
    """Apply a serialized component.

    :param path: Path to the stack.
    """
    db = superduper()
    component = Component.read(path)
    db.apply(component)
