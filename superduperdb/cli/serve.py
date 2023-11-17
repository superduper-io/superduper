import subprocess
import typing as t

import superduperdb as s

from . import command


@command(help='Start server')
def serve():
    from superduperdb.base.build import build_datalayer
    from superduperdb.server.server import serve

    db = build_datalayer()
    serve(db)


@command(help='Start local dask scheduler')
def dask_scheduler():
    subprocess.Popen(["dask", "scheduler"])


@command(help='Start local dask worker')
def dask_worker():
    subprocess.Popen(["dask", "worker", "tcp://localhost:8786"])


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


@command(help='Start vector search server')
def vector_search():
    from superduperdb.vector_search.server.app import app

    app.start()


@command(help='Start standalone change data capture')
def cdc():
    from superduperdb.cdc.app import app

    app.start()
