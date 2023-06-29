from . import command
from superduperdb.core.suri import URIDocument
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
    server = Server(document_store=URIDocument.cache)
    db = build.build_datalayer()

    server.register(db.delete)
    server.register(db.insert)
    server.register(db.select)
    server.register(db.update)
    server.register(db.execute)

    if open_page:

        def target():
            time.sleep(open_delay)
            cfg = server.cfg.web_server
            url = f'http://{cfg.host}:{cfg.port}'

            webbrowser.open(url, new=1, autoraise=True)

        Thread(target=target, daemon=True).start()

    server.run(db)
