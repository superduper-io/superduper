import typing as t

from superduperdb import CFG
from superduperdb.datalayer.base.backends import (
    data_backends,
    metadata_stores,
    artifact_stores,
    connections as default_connections,
)
from superduperdb.datalayer.base.database import BaseDatabase

from superduperdb.datalayer.mongodb.artifacts import MongoArtifactStore
from superduperdb.datalayer.mongodb.data_backend import MongoDataBackend
from superduperdb.datalayer.mongodb.metadata import MongoMetaDataStore
from superduperdb.misc.config import DataLayer

TBuild = t.TypeVar(
    'TBuild',
    t.Type[MongoArtifactStore],
    t.Type[MongoDataBackend],
    t.Type[MongoMetaDataStore],
)


def build_datalayer(**connections: t.Any) -> BaseDatabase:
    def build(
        cfg: DataLayer,
        stores: t.Dict[str, TBuild],
    ) -> t.Any:
        cls = stores[cfg.cls]
        if connections:
            connection = connections[cfg.connection]
        else:
            connection = default_connections[cfg.connection](**cfg.kwargs)

        return cls(name=cfg.name, conn=connection)

    return BaseDatabase(
        artifact_store=build(CFG.data_layers.artifact, artifact_stores),
        db=build(CFG.data_layers.data_backend, data_backends),
        metadata=build(CFG.data_layers.metadata, metadata_stores),
    )
