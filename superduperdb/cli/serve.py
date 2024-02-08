from . import command


@command(help='Start local cluster')
def local_cluster():
    from superduperdb.base.build import build_datalayer
    from superduperdb.server.cluster import cluster

    db = build_datalayer()
    cluster(db)


@command(help='Start vector search server')
def vector_search():
    from superduperdb.vector_search.server.app import app

    app.start()


@command(help='Start standalone change data capture')
def cdc():
    from superduperdb.cdc.app import app

    app.start()
