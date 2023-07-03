from . import command
from superduperdb.datalayer.base import build
from superduperdb.server.make import make_server
from threading import Thread
import superduperdb as s
import time
import webbrowser


@command(help='Start server')
def serve(
    open_page: bool = True,
    open_delay: float = 0.5,
):
    db = build.build_datalayer()
    server = make_server(db)

    if open_page:
        Thread(target=_open_page, args=(open_delay,), daemon=True).start()

    server.run(db)


def _open_page(open_delay):
    time.sleep(open_delay)
    cfg = s.CFG.server.web_server
    url = f'http://{cfg.host}:{cfg.port}'

    webbrowser.open(url, new=1, autoraise=True)
