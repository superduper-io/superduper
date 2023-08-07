import inspect

import superduperdb as s
from superduperdb.db.base.backends import artifact_stores
from superduperdb.db.base.backends import connections as default_connections
from superduperdb.db.base.backends import (
    data_backends,
    metadata_stores,
    vector_database_stores,
)
from superduperdb.db.base.db import DB
from superduperdb.server.dask_client import dask_client


def build_vector_database(cfg):
    """
    Build vector database as per ``vector_database = DB.vector_database``
    from configuration.

    :param cfg: configuration to use. (See ``superduperdb.CFG.vector_search``)
    """
    cls = vector_database_stores[cfg.__class__]
    sig = inspect.signature(cls.__init__)
    kwargs = {k: v for k, v in cfg.dict().items() if k in sig.parameters}
    return cls(**kwargs)


def build_datalayer(cfg=None, **connections) -> DB:
    """
    Build db as per ``db = superduper(db)`` from configuration.

    :param cfg: configuration to use. If None, use ``superduperdb.CFG``.
    :param connections: cache of connections to reuse in the build process.
    """
    cfg = cfg or s.CFG

    def build_distributed_client(cfg):
        if cfg.distributed:
            return dask_client(cfg.dask)

    def build(cfg, stores):
        cls = stores[cfg.cls]
        if connections:
            connection = connections[cfg.connection]
        else:
            # cast port to an integer.
            cfg.kwargs['port'] = int(cfg.kwargs['port'])
            connection = default_connections[cfg.connection](**cfg.kwargs)

        return cls(name=cfg.name, conn=connection)

    return DB(
        artifact_store=build(cfg.data_layers.artifact, artifact_stores),
        databackend=build(cfg.data_layers.data_backend, data_backends),
        metadata=build(cfg.data_layers.metadata, metadata_stores),
        vector_database=build_vector_database(cfg.vector_search.type),
        distributed_client=build_distributed_client(cfg),
    )
