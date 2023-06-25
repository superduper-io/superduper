from . import command
from superduperdb.datalayer.base import build
from superduperdb.server.server import Server
from threading import Thread
import time
import webbrowser


@command(help='Start server')
def serve(
    open_page: bool = True,
    open_delay: float = 0.5,
):
    s = Server()
    db = build.build_datalayer()

    s.register(db.select)
    s.register(db.delete)

    if open_page:

        def target():
            time.sleep(open_delay)
            cfg = s.cfg.web_server
            url = f'http://{cfg.host}:{cfg.port}'

            webbrowser.open(url, new=1, autoraise=True)

        Thread(target=target, daemon=True).start()
    s.run(db)
