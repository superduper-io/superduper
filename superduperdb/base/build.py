import re

import superduperdb as s
from superduperdb.backends.base.backends import data_backends, metadata_stores
from superduperdb.backends.filesystem.artifacts import FileSystemArtifactStore
from superduperdb.backends.mongodb.artifacts import MongoArtifactStore
from superduperdb.base.datalayer import Datalayer
from superduperdb.server.dask_client import dask_client


def build_artifact_store(cfg):
    if cfg.artifact_store is None:
        raise ValueError('No artifact store specified')
    elif cfg.artifact_store.startswith('mongodb://'):
        import pymongo

        conn = pymongo.MongoClient('/'.join(cfg.artifact_store.split('/')[:-1]))
        name = cfg.artifact_store.split('/')[-1]
        return MongoArtifactStore(conn, name)
    elif cfg.artifact_store.startswith('filesystem://'):
        directory = cfg.artifact_store.split('://')[1]
        return FileSystemArtifactStore(directory)
    else:
        raise ValueError(f'Unknown artifact store: {cfg.artifact_store}')


def build_datalayer(cfg=None, **kwargs) -> Datalayer:
    """
    Build db as per ``db = superduper(db)`` from configuration.

    :param cfg: configuration to use. If None, use ``superduperdb.CFG``.
    """
    cfg = cfg or s.CFG

    for k, v in kwargs.items():
        setattr(cfg, k, v)

    def build(uri, mapping):
        if re.match('^mongodb:\/\/|^mongodb\+srv:\/\/', uri) is not None:
            name = uri.split('/')[-1]
            import pymongo

            uri = '/'.join(uri.split('/')[:-1])
            conn = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
            return mapping['mongodb'](conn, name)
        elif uri.startswith('mongomock://'):
            name = uri.split('/')[-1]
            import mongomock

            conn = mongomock.MongoClient()
            return mapping['mongodb'](conn, name)
        else:
            import ibis

            conn = ibis.connect(uri)
            return mapping['ibis'](conn, uri.split('//')[0])

    databackend = build(cfg.data_backend, data_backends)

    db = Datalayer(
        databackend=databackend,
        metadata=(
            build(cfg.metadata_store, metadata_stores)
            if cfg.metadata_store is not None
            else databackend.build_metadata()
        ),
        artifact_store=(
            build_artifact_store(cfg)
            if cfg.artifact_store is not None
            else databackend.build_artifact_store()
        ),
        distributed_client=dask_client(
            cfg.cluster.dask_scheduler,
            local=cfg.cluster.local,
            serializers=cfg.cluster.serializers,
            deserializers=cfg.cluster.deserializers,
        )
        if cfg.cluster.distributed
        else None,
    )

    return db
