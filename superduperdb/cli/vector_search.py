from superduperdb.vector_search.server.app import app

from . import command


@command(help='Start vector search server')
def vector_search():
    app.start()
