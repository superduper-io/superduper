import typing as t

import superduperdb as s

from . import command


@command(help='Start server')
def serve():
    from superduperdb.base.build import build_datalayer
    from superduperdb.server.server import serve

    db = build_datalayer()
    serve(db)


@command(help='Start local dask cluster')
def local_dask():
    raise NotImplementedError


@command(help='Start standalone change data capture')
def cdc():
    from superduperdb.base.build import build_datalayer

    db = build_datalayer()
    db.cdc.listen()


@command(help='Start local cluster: server, dask and change data capture')
def local_cluster(on: t.List[str] = []):
    from superduperdb.backends.mongodb.query import Collection
    from superduperdb.base.build import build_datalayer
    from superduperdb.server.dask_client import dask_client
    from superduperdb.server.server import serve

    db = build_datalayer()
    dask_client(
        uri=s.CFG.cluster.dask_scheduler,
        local=True,
    )
    for collection in on:
        db.cdc.listen(
            on=Collection(identifier=collection),
        )
    serve(db)
