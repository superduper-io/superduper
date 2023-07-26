from . import command
import typing as t

from superduperdb.datalayer.base.build import build_datalayer
from superduperdb.datalayer.base.cdc import DatabaseWatcher
from superduperdb.datalayer.mongodb.query import Collection
from superduperdb.cluster.server import serve as _serve
from superduperdb.cluster.dask_client import dask_client
from superduperdb import CFG


@command(help='Start server')
def serve():
    db = build_datalayer()
    _serve(db)


@command(help='Start local cluster: server, dask and change data capture')
def local_cluster(on: t.List[str] = ()):
    db = build_datalayer()
    dask_client(CFG.dask, local=True)
    for collection in on:
        w = DatabaseWatcher(
            db=db,
            on=Collection(name=collection),
        )
        w.watch()
    _serve(db)
