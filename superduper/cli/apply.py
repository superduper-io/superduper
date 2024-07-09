from superduper import Component, logging, superduper

from . import command


@command(help='Apply the component to a `superduper` deployment')
def apply(path: str):
    """Apply a serialized component.

    :param path: Path to the stack.
    """
    _apply(path)


@command(help='`superduper` deployment')
def drop(data: bool = False, force: bool = False):
    """Apply a serialized component.

    :param path: Path to the stack.
    """
    db = superduper()
    db.drop(force=force, data=data)
    db.disconnect()


def _apply(path: str):
    try:
        logging.info('Connecting to superduper')
        db = superduper()
        logging.info('Reading component')
        component = Component.read(path)
        logging.info('Applying component to superduper')
        db.apply(component)
    except Exception as e:
        raise e
    finally:
        db.disconnect()
