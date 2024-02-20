import os
import re
import sys
import typing as t

import ibis
import mongomock
import pandas
import pymongo

import superduperdb as s
from superduperdb import logging
from superduperdb.backends.base.backends import data_backends, metadata_stores
from superduperdb.backends.base.data_backend import BaseDataBackend
from superduperdb.backends.local.artifacts import FileSystemArtifactStore
from superduperdb.backends.local.compute import LocalComputeBackend
from superduperdb.backends.mongodb.artifacts import MongoArtifactStore
from superduperdb.backends.ray.compute import RayComputeBackend
from superduperdb.base.datalayer import Datalayer


def _build_metadata(cfg, databackend: t.Optional['BaseDataBackend'] = None):
    # Connect to metadata store.
    # ------------------------------
    # 1. try to connect to the metadata store specified in the configuration.
    # 2. if that fails, try to connect to the data backend engine.
    # 3. if that fails, try to connect to the data backend uri.
    if cfg.metadata_store is not None:
        # try to connect to the metadata store specified in the configuration.
        logging.info("Connecting to Metadata Client:", cfg.metadata_store)
        return _build_databackend_impl(
            cfg.metadata_store, metadata_stores, type='metadata'
        )
    else:
        try:
            # try to connect to the data backend engine.
            assert isinstance(databackend, BaseDataBackend)
            logging.info(
                "Connecting to Metadata Client with engine: ", databackend.conn
            )
            return databackend.build_metadata()
        except Exception as e:
            logging.warn("Error building metadata from DataBackend:", str(e))
            metadata = None

    if metadata is None:
        try:
            # try to connect to the data backend uri.
            logging.info("Connecting to Metadata Client with URI: ", cfg.data_backend)
            return _build_databackend_impl(
                cfg.data_backend, metadata_stores, type='metadata'
            )
        except Exception as e:
            # Exit quickly if a connection fails.
            logging.error("Error initializing to Metadata Client:", str(e))
            sys.exit(1)


def _build_databackend(cfg, databackend=None):
    # Connect to data backend.
    # ------------------------------
    try:
        if not databackend:
            databackend = _build_databackend_impl(cfg.data_backend, data_backends)
        logging.info("Data Client is ready.", databackend.conn)
    except Exception as e:
        # Exit quickly if a connection fails.
        logging.error("Error initializing to DataBackend Client:", str(e))
        sys.exit(1)
    return databackend


def _build_artifact_store(
    artifact_store: t.Optional[str] = None,
    databackend: t.Optional['BaseDataBackend'] = None,
):
    if not artifact_store:
        assert isinstance(databackend, BaseDataBackend)
        return databackend.build_artifact_store()

    if artifact_store.startswith('mongodb://'):
        import pymongo

        conn: pymongo.MongoClient = pymongo.MongoClient(
            '/'.join(artifact_store.split('/')[:-1])
        )
        name = artifact_store.split('/')[-1]
        return MongoArtifactStore(conn, name)
    elif artifact_store.startswith('filesystem://'):
        directory = artifact_store.split('://')[1]
        return FileSystemArtifactStore(directory)
    else:
        raise ValueError(f'Unknown artifact store: {artifact_store}')


# Helper function to build a data backend based on the URI.
def _build_databackend_impl(uri, mapping, type: str = 'data_backend'):
    logging.debug(f"Parsing data connection URI:{uri}")

    if re.match('^mongodb:\/\/', uri) is not None:
        name = uri.split('/')[-1]
        conn: pymongo.MongoClient = pymongo.MongoClient(
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

    elif uri.endswith('.csv'):
        if type == 'metadata':
            raise ValueError('Cannot build metadata from a CSV file.')

        import glob

        csv_files = glob.glob(uri)
        tables = {}
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            tables[filename] = pandas.read_csv(csv_file)
        ibis_conn = ibis.pandas.connect(tables)
        return mapping['ibis'](ibis_conn, uri.split('/')[0])
    else:
        name = uri.split('//')[0]
        if type == 'data_backend':
            ibis_conn = ibis.connect(uri)
            return mapping['ibis'](ibis_conn, name)
        else:
            assert type == 'metadata'
            from sqlalchemy import create_engine

            sql_conn = create_engine(uri)
            return mapping['sqlalchemy'](sql_conn, name)


def _build_compute(compute):
    logging.info("Connecting to compute client:", compute)

    if compute == 'local' or compute is None:
        return LocalComputeBackend()

    if compute == 'dask+thread':
        from superduperdb.backends.dask.compute import DaskComputeBackend

        return DaskComputeBackend('local', local=True)

    if compute.split('://')[0] == 'dask+tcp':
        from superduperdb.backends.dask.compute import DaskComputeBackend

        uri = compute.split('+')[-1]
        return DaskComputeBackend(uri)

    if compute.split('://')[0] == 'ray':
        return RayComputeBackend(compute)

    raise ValueError('Compute {compute} is not a valid compute configuration.')


def build_datalayer(cfg=None, databackend=None, **kwargs) -> Datalayer:
    """
    Build a Datalayer object as per ``db = superduper(db)`` from configuration.

    :param cfg: Configuration to use. If None, use ``superduperdb.CFG``.
    :param databackend: Databacked to use.
                        If None, use ``superduperdb.CFG.data_backend``.
    :pararm kwargs: keyword arguments to be adopted by the `CFG`
    """

    # Configuration
    # ------------------------------
    # Use the provided configuration or fall back to the default configuration.
    cfg = cfg or s.CFG

    # Update configuration with keyword arguments.
    for k, v in kwargs.items():
        if '__' in k:
            getattr(cfg, k.split('__')[0]).force_set(k.split('__')[1], v)
        else:
            cfg.force_set(k, v)

    # Build databackend
    databackend = _build_databackend(cfg, databackend)

    # Build metadata store
    metadata = _build_metadata(cfg, databackend)
    assert metadata

    # Build artifact store
    artifact_store = _build_artifact_store(cfg.artifact_store, databackend)

    # Build compute
    compute = _build_compute(cfg.cluster.compute)

    # Build DataLayer
    # ------------------------------
    db = Datalayer(
        databackend=databackend,
        metadata=metadata,
        artifact_store=artifact_store,
        compute=compute,
    )

    return db
