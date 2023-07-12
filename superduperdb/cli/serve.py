from . import command
from superduperdb.serve.server import serve as _serve
import superduperdb as s
import time
import webbrowser


@command(help='Start server')
def serve():
    _serve()


def _open_page(open_delay):
    time.sleep(open_delay)
    cfg = s.CFG.server.web_server
    url = f'http://{cfg.host}:{cfg.port}'
    webbrowser.open(url, new=1, autoraise=True)
