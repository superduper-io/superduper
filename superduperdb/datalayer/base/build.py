from superduperdb.datalayer.base.backends import (
    data_backends,
    metadata_stores,
    artifact_stores,
    vector_database_stores,
    connections as default_connections,
)
from superduperdb.datalayer.base.database import BaseDatabase
from superduperdb.cluster.dask.dask_client import dask_client


def build_datalayer(cfg=None, **connections) -> BaseDatabase:
    """
    Build datalayer as per ``db = superduper(db)`` from configuration.

    :param connections: cache of connections to reuse in the build process.
    """
    if not cfg:
        from superduperdb import CFG
    else:
        CFG = cfg

    def build_vector_database(cfg, stores):
        cls = stores[cfg.__class__]
        return cls(cfg)

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

    return BaseDatabase(
        artifact_store=build(CFG.data_layers.artifact, artifact_stores),
        databackend=build(CFG.data_layers.data_backend, data_backends),
        metadata=build(CFG.data_layers.metadata, metadata_stores),
        vector_database=build_vector_database(
            CFG.vector_search.type, vector_database_stores
        ),
        distributed_client=build_distributed_client(CFG),
    )
