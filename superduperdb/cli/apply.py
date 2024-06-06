from superduperdb import Component, logging, superduper

from . import command


@command(help='Apply the component to a `superduperdb` deployment')
def apply(path: str):
    """Apply a serialized component.

    :param path: Path to the stack.
    """
    _apply(path)


@command(help='`superduperdb` deployment')
def drop(data: bool = False):
    """Apply a serialized component.

    :param path: Path to the stack.
    """
    db = superduper()
    db.drop(data=data)
    db.disconnect()


def _apply(path: str):
    try:
        logging.info('Connecting to superduperdb')
        db = superduper()
        logging.info('Reading component')
        component = Component.read(path)
        logging.info('Applying component to superduperdb')
        db.apply(component)
    except Exception as e:
        raise e
    finally:
        db.disconnect()
