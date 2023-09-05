import typing as t

import superduperdb as s

from . import command


@command(help='Start server')
def serve():
    from superduperdb.db.base.build import build_datalayer
    from superduperdb.server.server import serve

    db = build_datalayer()
    serve(db)


@command(help='Start local cluster: server, dask and change data capture')
def local_cluster(on: t.List[str] = []):
    from superduperdb.db.base.build import build_datalayer
    from superduperdb.db.base.cdc import DatabaseListener
    from superduperdb.db.mongodb.query import Collection
    from superduperdb.server.dask_client import dask_client
    from superduperdb.server.server import serve

    db = build_datalayer()
    dask_client(
        uri=s.CFG.cluster.dask_scheduler,
        local=True,
    )
    for collection in on:
        w = DatabaseListener(
            db=db,
            on=Collection(name=collection),
        )
        w.listen()
    serve(db)
