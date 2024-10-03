import typing as t
import json
import os

from superduper import Component, logging, superduper
from superduper.components.template import Template

from . import command


@command(help='Apply a template or application to a `superduper` deployment')
def apply(name: str, variables: str | None = None):
    """Apply a serialized component.

    :param name: Path or name of the template/ component.
    :param values: JSON string of values to apply to the template.
    """
    _apply(name, variables)


@command(help='`superduper` deployment')
def drop(data: bool = False, force: bool = False):
    """Apply a serialized component.

    :param path: Path to the stack.
    """
    db = superduper()
    db.drop(force=force, data=data)
    db.disconnect()


def _apply(name: str, variables: str | None = None):

    def _build_from_template(t):
        assert variables is not None, 'Variables must be provided for templates'
        loaded = json.loads(variables)
        return t(**loaded)

    if os.path.exists(name):
        try:
            t = Template.read(name)
            c = _build_from_template(t)
        except Exception as e:
            if 'Expecting' in str(e):
                c = Component.read(name)
    else:
        from superduper import templates
        t = getattr(templates, name)
        c = _build_from_template(t)

    try:
        logging.info('Connecting to superduper')
        db = superduper()
        db.apply(c)
    except Exception as e:
        raise e
    finally:
        db.disconnect()
