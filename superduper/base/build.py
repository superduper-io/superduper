import re
import typing as t

from prettytable import PrettyTable

import superduper as s
from superduper import logging
from superduper.backends.base.backends import data_backends, metadata_stores
from superduper.backends.base.data_backend import BaseDataBackend, DataBackendProxy
from superduper.backends.base.metadata import MetaDataStoreProxy
from superduper.backends.local.artifacts import FileSystemArtifactStore
from superduper.backends.mongodb.artifacts import MongoArtifactStore
from superduper.base.datalayer import Datalayer
from superduper.misc.anonymize import anonymize_url


def _get_metadata_store(cfg):
    # try to connect to the metadata store specified in the configuration.
    logging.info("Connecting to Metadata Client:", cfg.metadata_store)
    return _build_databackend_impl(cfg.metadata_store, metadata_stores, type='metadata')


def _build_metadata(cfg, databackend: t.Optional['BaseDataBackend'] = None):
    # Connect to metadata store.
    # ------------------------------
    # 1. try to connect to the metadata store specified in the configuration.
    # 2. if that fails, try to connect to the data backend engine.
    # 3. if that fails, try to connect to the data backend uri.
    if cfg.metadata_store is not None:
        return _get_metadata_store(cfg)
    else:
        try:
            # try to connect to the data backend engine.
            assert isinstance(databackend, DataBackendProxy)
            logging.info(
                "Connecting to Metadata Client with engine: ", databackend.conn
            )
            return databackend.build_metadata()
        except Exception as e:
            logging.warn("Error building metadata from DataBackend:", str(e))
            metadata = None

    if metadata is None:
        # try to connect to the data backend uri.
        logging.info("Connecting to Metadata Client with URI: ", cfg.data_backend)
        return _build_databackend_impl(
            cfg.data_backend, metadata_stores, type='metadata'
        )


def _build_databackend(cfg, databackend=None):
    # Connect to data backend.
    # ------------------------------
    if not databackend:
        databackend = _build_databackend_impl(cfg.data_backend, data_backends)
    logging.info("Data Client is ready.", databackend.conn)
    return databackend


def _build_artifact_store(
    artifact_store: t.Optional[str] = None,
    databackend: t.Optional['BaseDataBackend'] = None,
):
    if not artifact_store:
        assert isinstance(databackend, DataBackendProxy)
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


class _MetaDataMatcher:
    patterns = {
        '^mongodb:\/\/': ('mongodb', 'mongodb'),
        '^mongodb\+srv:\/\/': ('mongodb', 'atlas'),
        '^mongomock:\/\/': ('mongodb', 'mongomock'),
    }
    not_supported = [('sqlalchemy', 'pandas')]

    @classmethod
    def create(cls, uri, mapping: t.Dict):
        """Helper method to create metadata backend."""
        backend = 'sqlalchemy'
        flavour = 'base'
        for pattern in cls.patterns:
            if re.match(pattern, uri) is not None:
                backend, flavour = cls.patterns[pattern]
                if (backend, flavour) in cls.not_supported:
                    raise ValueError(
                        f"{backend} with flavour {flavour} not supported "
                        "to create metadata store."
                    )
                return mapping[backend](uri, flavour=flavour)

        return mapping[backend](uri)


class _DataBackendMatcher(_MetaDataMatcher):
    patterns = {**_MetaDataMatcher.patterns, r'.*\.csv$': ('ibis', 'pandas')}

    @classmethod
    def create(cls, uri, mapping: t.Dict):
        """Helper method to create databackend."""
        backend = 'ibis'
        for pattern in cls.patterns:
            if re.match(pattern, uri) is not None:
                backend, flavour = cls.patterns[pattern]

                return mapping[backend](uri, flavour=flavour)

        return mapping[backend](uri, flavour='base')


# Helper function to build a data backend based on the URI.
def _build_databackend_impl(uri, mapping, type: str = 'data_backend'):
    logging.debug(f"Parsing data connection URI:{uri}")
    if type == 'data_backend':
        db = DataBackendProxy(_DataBackendMatcher.create(uri, mapping))
    else:
        db = MetaDataStoreProxy(_MetaDataMatcher.create(uri, mapping))
    return db


def build_compute(compute):
    """
    Helper function to build compute backend.

    :param compute: Compute uri.
    """
    logging.info("Connecting to compute client:", compute)
    path = compute._path or 'superduper.backends.local.compute.LocalComputeBackend'
    spath = path.split('.')
    path, cls = '.'.join(spath[:-1]), spath[-1]

    import importlib

    module = importlib.import_module(path)
    compute_cls = getattr(module, cls)

    return compute_cls(compute.uri)


def build_datalayer(cfg=None, databackend=None, **kwargs) -> Datalayer:
    """
    Build a Datalayer object as per ``db = superduper(db)`` from configuration.

    :param cfg: Configuration to use. If None, use ``superduper.CFG``.
    :param databackend: Databacked to use.
                        If None, use ``superduper.CFG.data_backend``.
    :param kwargs: keyword arguments to be adopted by the `CFG`
    """
    # Configuration
    # ------------------------------
    # Use the provided configuration or fall back to the default configuration.
    cfg = (cfg or s.CFG)(**kwargs)

    databackend = _build_databackend(cfg, databackend)
    metadata = _build_metadata(cfg, databackend)
    assert metadata

    artifact_store = _build_artifact_store(cfg.artifact_store, databackend)
    compute = build_compute(cfg.cluster.compute)

    datalayer = Datalayer(
        databackend=databackend,
        metadata=metadata,
        artifact_store=artifact_store,
        compute=compute,
    )
    # Keep the real configuration in the datalayer object.
    datalayer.cfg = cfg

    show_configuration(cfg)
    return datalayer


def show_configuration(cfg):
    """Show the configuration.

    Only show the important configuration values and anonymize the URLs.

    :param cfg: The configuration object.
    """
    table = PrettyTable()
    table.field_names = ["Configuration", "Value"]
    # Only show the important configuration values.
    key_values = [
        ('Data Backend', anonymize_url(cfg.data_backend)),
        ('Metadata Store', anonymize_url(cfg.metadata_store)),
        ('Artifact Store', anonymize_url(cfg.artifact_store)),
        ('Compute', cfg.cluster.compute.uri),
    ]
    for key, value in key_values:
        if value:
            table.add_row([key, value])

    logging.info(f"Configuration: \n {table}")
