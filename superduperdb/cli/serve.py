import typing as t

from superduperdb import CFG
from superduperdb.db.base.build import build_datalayer
from superduperdb.db.base.cdc import DatabaseWatcher
from superduperdb.db.mongodb.query import Collection
from superduperdb.server.dask_client import dask_client
from superduperdb.server.server import serve as _serve

from . import command


@command(help='Start server')
def serve():
    db = build_datalayer()
    _serve(db)


@command(help='Start local cluster: server, dask and change data capture')
def local_cluster(on: t.List[str] = []):
    db = build_datalayer()
    dask_client(CFG.dask, local=True)
    for collection in on:
        w = DatabaseWatcher(
            db=db,
            on=Collection(name=collection),
        )
        w.watch()
    _serve(db)
