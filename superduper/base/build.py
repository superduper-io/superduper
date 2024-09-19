import re
import typing as t

from prettytable import PrettyTable

import superduper as s
from superduper import logging
from superduper.backends.base.cluster import Cluster
from superduper.backends.base.data_backend import DataBackendProxy
from superduper.backends.base.metadata import MetaDataStoreProxy
from superduper.base.config import Config
from superduper.base.datalayer import Datalayer
from superduper.misc.anonymize import anonymize_url
from superduper.misc.plugins import load_plugin


class _Loader:
    not_supported: t.Tuple = ()

    @classmethod
    def match(cls, uri):
        """Check if the uri matches the pattern."""
        plugin, flavour = None, None
        for pattern in cls.patterns:
            if re.match(pattern, uri) is not None:
                selection = cls.patterns[pattern]
                if isinstance(selection, tuple):
                    plugin, flavour = selection
                else:
                    assert isinstance(selection, str)
                    plugin = selection
                break
        if plugin is None:
            raise ValueError(f"{cls.__name__} No support for uri: {uri}")
        return plugin, flavour

    @classmethod
    def create(cls, uri):
        """Helper method to create metadata backend."""
        plugin, flavour = cls.match(uri)
        if cls.not_supported and (plugin, flavour) in cls.not_supported:
            raise ValueError(
                f"{plugin} with flavour {flavour} not supported "
                "to create metadata store."
            )
        impl = getattr(load_plugin(plugin), cls.impl)
        return impl(uri, flavour=flavour)


class _MetaDataLoader(_Loader):
    impl = 'MetaDataStore'
    patterns = {
        r'^mongodb:\/\/': ('mongodb', 'mongodb'),
        r'^mongodb\+srv:\/\/': ('mongodb', 'atlas'),
        r'^mongomock:\/\/': ('mongodb', 'mongomock'),
        r'^sqlite:\/\/': ('sqlalchemy', 'base'),
        r'^postgresql:\/\/': ('sqlalchemy', 'base'),
        r'^snowflake:\/\/': ('sqlalchemy', 'base'),
        r'^duckdb:\/\/': ('sqlalchemy', 'base'),
        r'^mssql:\/\/': ('sqlalchemy', 'base'),
        r'^mysql:\/\/': ('sqlalchemy', 'base'),
    }


class _DataBackendLoader(_Loader):
    impl = 'DataBackend'
    patterns = {
        r'^mongodb:\/\/': ('mongodb', 'mongodb'),
        r'^mongodb\+srv:\/\/': ('mongodb', 'atlas'),
        r'^mongomock:\/\/': ('mongodb', 'mongomock'),
        r'^sqlite://': ('ibis', 'base'),
        r'^postgresql://': ('ibis', 'base'),
        r'^snowflake:\/\/': ('ibis', 'base'),
        r'^duckdb://': ('ibis', 'base'),
        r'^mssql://': ('ibis', 'base'),
        r'^mysql://': ('ibis', 'base'),
        r'.*\*.csv$': ('ibis', 'pandas'),
    }


class _ArtifactStoreLoader(_Loader):
    impl = 'ArtifactStore'
    patterns = {
        r'^filesystem:\/\/': ('local', 'base'),
        r'^mongomock:\/\/': ('local', 'base'),
        r'^mongodb\+srv:\/\/': ('mongodb', 'atlas'),
        r'^mongodb:\/\/': ('mongodb', 'base'),
        r'^sqlite:\/\/': ('local', 'base'),
        r'^postgresql:\/\/': ('local', 'base'),
        r'^snowflake:\/\/': ('local', 'base'),
    }


def _build_artifact_store(uri):
    return _ArtifactStoreLoader.create(uri)


def _build_databackend(uri):
    return DataBackendProxy(_DataBackendLoader.create(uri))


def _build_metadata(uri):
    db = MetaDataStoreProxy(_MetaDataLoader.create(uri))
    return db


def _build_compute(cfg):
    """
    Helper function to build compute backend.

    :param cfg: SuperDuper config.
    """
    backend = getattr(load_plugin(cfg.cluster.compute.backend), 'ComputeBackend')
    return backend(uri=cfg.cluster.compute.uri, **cfg.cluster.compute.kwargs)


def build_datalayer(cfg=None, **kwargs) -> Datalayer:
    """
    Build a Datalayer object as per ``db = superduper(db)`` from configuration.

    :param cfg: Configuration to use. If None, use ``superduper.CFG``.
    :param kwargs: keyword arguments to be adopted by the `CFG`
    """
    # Configuration
    # ------------------------------
    # Use the provided configuration or fall back to the default configuration.
    cfg = (cfg or s.CFG)(**kwargs)
    cfg = t.cast(Config, cfg)
    databackend_obj = _build_databackend(cfg.data_backend)
    if cfg.metadata_store:
        metadata_obj = _build_metadata(cfg.metadata_store)
    else:
        metadata_obj = databackend_obj.build_metadata()

    if cfg.artifact_store:
        artifact_store = _build_artifact_store(cfg.artifact_store)
    else:
        artifact_store = databackend_obj.build_artifact_store()

    backend = getattr(load_plugin(cfg.cluster_engine), 'Cluster')    
    cluster = backend.build(cfg, **kwargs)

    datalayer = Datalayer(
        databackend=databackend_obj,
        metadata=metadata_obj,
        artifact_store=artifact_store,
        cluster=cluster
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
    ]
    for key, value in key_values:
        if value:
            table.add_row([key, value])

    logging.info(f"Configuration: \n {table}")
