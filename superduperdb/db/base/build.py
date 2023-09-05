import pymongo

import superduperdb as s
from superduperdb.base.logger import logging
from superduperdb.db.base.backends import (
    artifact_stores,
    data_backends,
    metadata_stores,
    vector_data_stores,
)
from superduperdb.db.base.db import DB
from superduperdb.server.dask_client import dask_client


def build_vector_database(cfg):
    """
    Build vector database as per ``vector_database = DB.vector_database``
    from configuration.

    :param cfg: configuration to use. (See ``superduperdb.CFG.vector_search``)
    """
    if cfg.vector_search == cfg.data_backend:
        logging.warning(
            'Vector database URI is the same as the data backend URI. '
            'Using the data backend as the vector database.'
        )
        return
    cls = vector_data_stores[cfg.vector_search.split('://')[0]]
    return cls(cfg.vector_search)


def build_datalayer(cfg=None) -> DB:
    """
    Build db as per ``db = superduper(db)`` from configuration.

    :param cfg: configuration to use. If None, use ``superduperdb.CFG``.
    """
    cfg = cfg or s.CFG

    def build(uri, mapping):
        if uri.startswith('mongodb://'):
            name = uri.split('/')[-1]
            uri = 'mongodb://' + uri.split('mongodb://')[-1].split('/')[0]
            conn = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
            return mapping['mongodb'](conn, name)
        else:
            import ibis

            conn = ibis.connect(uri)
            return mapping['ibis'](conn, uri.split('//')[0])

    databackend = build(cfg.data_backend, data_backends)

    logging.warn(cfg.data_backend)

    db = DB(
        databackend=databackend,
        metadata=(
            build(cfg.metadata_store, metadata_stores)
            if cfg.metadata_store is not None
            else databackend.build_metadata()
        ),
        artifact_store=(
            build(cfg.artifact_store, artifact_stores)
            if cfg.artifact_store is not None
            else databackend.build_artifact_store()
        ),
        vector_database=build_vector_database(cfg),
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
