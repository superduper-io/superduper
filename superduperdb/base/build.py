import re
import sys

import ibis
import mongomock
import pymongo

import superduperdb as s
from superduperdb import logging
from superduperdb.backends.base.backends import data_backends, metadata_stores
from superduperdb.backends.dask.compute import DaskComputeBackend
from superduperdb.backends.local.artifacts import FileSystemArtifactStore
from superduperdb.backends.local.compute import LocalComputeBackend
from superduperdb.backends.mongodb.artifacts import MongoArtifactStore
from superduperdb.base.datalayer import Datalayer


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


# Helper function to build a data backend based on the URI.
def build(uri, mapping):
    logging.debug(f"Parsing data connection URI:{uri}")

    if re.match('^mongodb:\/\/', uri) is not None:
        name = uri.split('/')[-1]
        conn = pymongo.MongoClient(
            uri,
            serverSelectionTimeoutMS=5000,
        )
        return mapping['mongodb'](conn, name)

    elif re.match('^mongodb\+srv:\/\/', uri):
        name = uri.split('/')[-1]
        conn = pymongo.MongoClient(
            '/'.join(uri.split('/')[:-1]),
            serverSelectionTimeoutMS=5000,
        )
        return mapping['mongodb'](conn, name)
    elif uri.startswith('mongomock://'):
        name = uri.split('/')[-1]
        conn = mongomock.MongoClient()
        return mapping['mongodb'](conn, name)
    else:
        conn = ibis.connect(uri)
        db_name = conn.current_database
        if 'ibis' in mapping:
            cls_ = mapping['ibis']
        elif 'sqlalchemy' in mapping:
            cls_ = mapping['sqlalchemy']
            conn = conn.con
        else:
            raise ValueError('No ibis or sqlalchemy backend specified')

        # if ':memory:' in uri:
        #     conn = uri
        #     db_name = None

        return cls_(conn, db_name)


def build_compute(cfg):
    compute = cfg.cluster.compute
    if compute == 'local' or compute is None:
        return LocalComputeBackend()

    if compute == 'dask+thread':
        return DaskComputeBackend('', local=True)

    if compute.split('://')[0] == 'dask+tcp':
        uri = compute.split('+')[-1]
        return DaskComputeBackend(uri)

    return LocalComputeBackend()


def build_datalayer(cfg=None, **kwargs) -> Datalayer:
    """
    Build a Datalayer object as per ``db = superduper(db)`` from configuration.

    :param cfg: Configuration to use. If None, use ``superduperdb.CFG``.
    """

    # Configuration
    # ------------------------------
    # Use the provided configuration or fall back to the default configuration.
    cfg = cfg or s.CFG

    # Update configuration with keyword arguments.
    for k, v in kwargs.items():
        cfg.force_set(k, v)

    # Connect to data backend.
    # ------------------------------
    try:
        databackend = build(cfg.data_backend, data_backends)
        logging.info("Data Client is ready.", databackend.conn)
    except Exception as e:
        # Exit quickly if a connection fails.
        logging.error("Error initializing to DataBackend Client:", str(e))
        sys.exit(1)

    # Build DataLayer
    # ------------------------------
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
        compute=build_compute(cfg),
    )

    return db
