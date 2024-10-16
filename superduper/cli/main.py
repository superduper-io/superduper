import json
import os
import typing as t

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


@command(help='Start rest server and user interface')
def start(
    port: int = 8000,
    remote_port: int = 8000,
    host: str = 'localhost',
    headless: bool = False,
):
    """Start the rest server and user interface.

    :param port: Port to run the server on.
    :param remote_port: Port to connect to remotely
    :param host: Host to connect to remotely
    :param headless: Toggle to ``True`` to suppress the browser.
    """
    from superduper.rest.base import SuperDuperApp
    from superduper.rest.build import build_frontend, build_rest_app

    app = SuperDuperApp('rest', port=remote_port)

    if host == 'localhost':
        # host frontend and server together
        assert remote_port == port
        build_rest_app(app)
        app.add_default_endpoints()
    else:
        logging.warn('Frontend pointing to remote server!')

    if not headless:
        build_frontend(app, port=remote_port, host=host)

    app.start()


@command(help='Initialize a template in the system')
def bootstrap(templates: t.List[str] | None = None):
    """Initialize a template in the system.

    :param templates: List of templates to initialize.
    """
    from superduper import templates as inbuilt

    if templates is None:
        templates = inbuilt.ls()
        templates = ['rag', 'text_vector_search']
    db = superduper()
    existing = db.show('template')
    for tem in templates:
        if tem in existing:
            logging.info(f'Template {tem} already exists')
            continue
        logging.info(f'Applying template: {tem} from inbuilt')
        tem = getattr(inbuilt, tem)
        db.apply(tem, force=True)


@command(help='Apply a template or application to a `superduper` deployment')
def ls():
    """Apply a serialized component.

    :param name: Path or name of the template/ component.
    :param values: JSON string of values to apply to the template.
    """
    from superduper.templates import ls

    for r in ls():
        print(r)


@command(help='Show available components')
def show(
    type_id: str | None = None,
    identifier: str | None = None,
    version: int | None = None,
):
    """Apply a serialized component.

    :param name: Path or name of the template/ component.
    :param values: JSON string of values to apply to the template.
    """
    db = superduper()
    to_show = db.show(type_id=type_id, identifier=identifier, version=version)
    import json

    logging.info('Showing components in system:')
    print(json.dumps(to_show, indent=2))


@command(help='`superduper` deployment')
def drop(data: bool = False, force: bool = False):
    """Apply a serialized component.

    :param path: Path to the stack.
    """
    db = superduper()
    db.drop(force=force, data=data)
    db.disconnect()


def _apply(name: str, variables: str | None = None):
    variables = variables or '{}'
    variables = json.loads(variables)

    def _build_from_template(t):
        assert variables is not None, 'Variables must be provided for templates'
        all_values = variables.copy()
        for k in t.template_variables:
            if k not in all_values:
                assert k in t.default_values, f'Variable {k} not specified'
                all_values[k] = t.default_values[k]
        return t(**all_values)

    db = superduper()

    if os.path.exists(name):
        with open(name + '/component.json', 'r') as f:
            info = json.load(f)
        if info['type_id'] == 'template':
            t = Template.read(name)
            c = _build_from_template(t)
        else:
            c = Component.read(name)
    else:
        existing = db.show('template')
        if name not in existing:
            from superduper import templates

            try:
                t = getattr(templates, name)
            except AttributeError:
                raise Exception(f'No pre-built template found of that name: {name}')
        else:
            t = db.load('template', name)

        c = _build_from_template(t)

    try:
        logging.info('Connecting to superduper')
        db = superduper()
        db.apply(c)
    except Exception as e:
        raise e
    finally:
        db.disconnect()
