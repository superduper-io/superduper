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


@command(help='Start rest server and user interface')
def start(
    port: int = 8000,
    remote_port: int = 8000,
    host: str = 'localhost',
    headless: bool = False,
    templates: bool = True,
):
    """Start the rest server and user interface.

    :param port: Port to run the server on.
    :param remote_port: Port to connect to remotely
    :param host: Host to connect to remotely
    :param headless: Toggle to ``True`` to suppress the browser.
    :param templates: Toggle to ``False`` to suppress initializing templates.
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

    build_frontend(app, port=remote_port, host=host)

    if headless:
        app.start()
    else:
        import threading
        import time
        import webbrowser

        server_thread = threading.Thread(target=lambda: app.start())
        server_thread.start()
        logging.info('Waiting for server to start')

        time.sleep(3)

    if templates:
        from superduper import templates

        db = app.app.state.pool
        existing = db.show('template')
        prebuilt = templates.ls()
        for t in prebuilt:
            if t not in existing:
                logging.info(f'Applying template \'{t}\'')
                db.apply(getattr(templates, t))

    if not headless:
        webbrowser.open(f'http://localhost:{port}')


@command(help='Apply a template or application to a `superduper` deployment')
def ls():
    """Apply a serialized component.

    :param name: Path or name of the template/ component.
    :param values: JSON string of values to apply to the template.
    """
    from superduper.templates import ls

    for r in ls():
        print(r)


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
