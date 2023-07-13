from superduperdb import CFG
from superduperdb.datalayer.base.backends import (
    data_backends,
    metadata_stores,
    artifact_stores,
    vector_database_stores,
    connections as default_connections,
)
from superduperdb.datalayer.base.database import BaseDatabase


def build_datalayer(**connections) -> BaseDatabase:
    """
    Build datalayer as per ``db = superduper(db)`` from configuration.

    :param connections: cache of connections to reuse in the build process.
    """

    def build_vector_database(cfg, stores):
        cls = stores[cfg.__class__]
        return cls(cfg)

    def build(cfg, stores):
        cls = stores[cfg.cls]
        if connections:
            connection = connections[cfg.connection]
        else:
            connection = default_connections[cfg.connection](**cfg.kwargs)

        return cls(name=cfg.name, conn=connection)

    return BaseDatabase(
        artifact_store=build(CFG.data_layers.artifact, artifact_stores),
        databackend=build(CFG.data_layers.data_backend, data_backends),
        metadata=build(CFG.data_layers.metadata, metadata_stores),
        vector_database=build_vector_database(
            CFG.vector_search.type, vector_database_stores
        ),
    )
