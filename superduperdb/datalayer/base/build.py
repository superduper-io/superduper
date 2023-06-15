from superduperdb import CFG
from superduperdb.datalayer.base.backends import (
    data_backends,
    metadata_stores,
    artifact_stores,
    connections as default_connections,
)


def build_datalayer(**connections):
    from superduperdb.datalayer.base.database import BaseDatabase

    data_backend_cls = data_backends[CFG.datalayer.data_backend_cls]
    metadata_store_cls = metadata_stores[CFG.datalayer.metadata_cls]
    artifact_store_cls = artifact_stores[CFG.datalayer.artifact_store_cls]

    if connections:
        data_backend_connection = connections[CFG.datalayer.data_backend_connection]
        metadata_store_connection = connections[CFG.datalayer.metadata_connection]
        artifact_store_connection = connections[CFG.datalayer.artifact_store_connection]
    else:
        data_backend_connection = default_connections[
            CFG.datalayer.data_backend_connection
        ](**CFG.datalayer.data_backend_kwargs)
        metadata_store_connection = default_connections[
            CFG.datalayer.metadata_connection
        ](**CFG.datalayer.metadata_kwargs)
        artifact_store_connection = default_connections[
            CFG.datalayer.artifact_store_connection
        ](**CFG.datalayer.artifact_store_kwargs)

    data_backend = data_backend_cls(
        name=CFG.datalayer.data_backend_name,
        conn=data_backend_connection,
    )
    artifact_store = artifact_store_cls(
        name=CFG.datalayer.artifact_store_name,
        conn=artifact_store_connection,
    )
    metadata = metadata_store_cls(
        name=CFG.datalayer.metadata_name,
        conn=metadata_store_connection,
    )

    return BaseDatabase(
        data_backend,
        metadata=metadata,
        artifact_store=artifact_store,
    )
