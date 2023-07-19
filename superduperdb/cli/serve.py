from . import command
import superduperdb as s
import time
import webbrowser


@command(help='Start server')
def serve():
    from superduperdb.serve.server import serve

    serve()


def _open_page(open_delay):
    time.sleep(open_delay)
    cfg = s.CFG.server.web_server
    url = f'http://{cfg.host}:{cfg.port}'
    webbrowser.open(url, new=1, autoraise=True)
