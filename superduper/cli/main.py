import json
import os

from superduper import CFG, Component, logging, superduper
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
    data_backend: str | None = None,
    templates: str | None = None,
):
    """Start the rest server and user interface.

    :param port: Port to run the server on.
    :param remote_port: Port to connect to remotely
    :param host: Host to connect to remotely
    :param headless: Toggle to ``True`` to suppress the browser.
    """
    from superduper import CFG
    from superduper.rest.base import SuperDuperApp
    from superduper.rest.build import build_frontend, build_rest_app

    CFG.log_colorize = False

    app = SuperDuperApp(
        'rest',
        port=remote_port,
        data_backend=data_backend,
        templates=templates.split(',') if templates else None,
    )

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


@command(help='Display a template')
def inspect(template: str):
    """Display a template."""
    root = os.path.dirname(os.path.dirname(__file__))
    from pygments import formatters, highlight, lexers

    path = f'{root}/templates/{template}/component.json'
    with open(path, 'r') as f:
        component = json.load(f)

    formatted_json = json.dumps(component, indent=2)
    colorful_json = highlight(
        formatted_json, lexers.JsonLexer(), formatters.TerminalFormatter()
    )
    print(colorful_json)


@command(help='Initialize a template in the system')
def bootstrap(
    template: str,
    destination: str | None = None,
    pip_install: bool = False,
    data_backend: str | None = None,
):
    """Initialize a template in the system.

    :param template: Template to initialize.

    Add template to the system.

    >>> superduper bootstrap rag

    Copy template to a directory to update it.

    >>> superduper bootstrap rag ./my_template
    """
    from superduper import templates as inbuilt

    data_backend = data_backend or CFG.data_backend

    db = superduper(data_backend)
    existing = db.show('template')

    if template.startswith('http'):
        import subprocess

        logging.info('Downloading remote template...')
        file_name = template.split('/')[-1]
        dir_name = file_name[:-4]
        subprocess.run(['curl', '-O', '-k', template])
        subprocess.run(['unzip', '-o', file_name, '-d', dir_name])
        template = dir_name

    if os.path.exists(template):
        if destination is not None:
            import shutil

            shutil.copytree(template, destination)
            return
        tem = Template.read(template)

    else:
        if destination is not None and os.path.exists(destination):
            tem = getattr(inbuilt, template)
            tem.export(destination)
            return

    if tem.identifier in existing:
        logging.warn(f'Template {tem.identifier} already exists')
        logging.warn('Aborting...')
        return

    logging.info(f'Installating template: {template}')

    if tem.requirements and pip_install:
        with open('/tmp/requirements.txt', 'w') as f:
            f.write('\n'.join(tem.requirements))
        subprocess.run(['pip', 'install', '-r', '/tmp/requirements.txt'])

    db.apply(tem)


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
    data_backend: str | None = None,
):
    """Apply a serialized component.

    :param name: Path or name of the template/ component.
    :param values: JSON string of values to apply to the template.
    """
    data_backend = data_backend or CFG.data_backend
    db = superduper(data_backend)
    to_show = db.show(type_id=type_id, identifier=identifier, version=version)
    import json

    logging.info('Showing components in system:')
    print(json.dumps(to_show, indent=2))


@command(help='Execute a query or prediction')
def execute(data_backend: str = None):
    """Execute a query or prediction."""
    from superduper.misc.interactive_prompt import _prompt

    _prompt(data_backend=data_backend)


@command(help='`superduper` deployment')
def drop(data: bool = False, force: bool = False, data_backend: str | None = None):
    """Drop the deployment.

    :param data: Drop the data.
    :param force: Force the drop.
    """
    data_backend = data_backend or CFG.data_backend
    db = superduper(data_backend)
    db.drop(force=force, data=data)
    db.disconnect()


def _apply(name: str, variables: str | None = None, data_backend: str | None = None):
    variables = variables or '{}'
    variables = json.loads(variables)

    # TODO remove all of this template logic
    def _build_from_template(t):
        assert variables is not None, 'Variables must be provided for templates'
        all_values = variables.copy()
        for k in t.template_variables:
            if k not in all_values:
                assert k in t.default_values, f'Variable {k} not specified'
                all_values[k] = t.default_values[k]
        return t(**all_values)

    data_backend = data_backend or CFG.data_backend
    db = superduper(data_backend)

    if os.path.exists(name):
        with open(name + '/component.json', 'r') as f:
            info = json.load(f)
        if info['type_id'] == 'template':
            t = Template.read(name)
            c = _build_from_template(t)
        else:
            c = Component.read(name)
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
