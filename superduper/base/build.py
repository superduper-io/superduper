import importlib
import re

from prettytable import PrettyTable

import superduper as s
from superduper import logging
from superduper.backends.base.data_backend import DataBackendProxy
from superduper.backends.base.metadata import MetaDataStoreProxy
from superduper.base.datalayer import Datalayer
from superduper.misc.anonymize import anonymize_url
from superduper.misc.plugins import load_plugin


class _Loader:
    @classmethod
    def create(cls, uri):
        """Helper method to create metadata backend."""
        backend = None
        flavour = None
        for pattern in cls.patterns:
            if re.match(pattern, uri) is not None:
                backend, flavour = cls.patterns[pattern]
                if (backend, flavour) in cls.not_supported:
                    raise ValueError(
                        f"{backend} with flavour {flavour} not supported "
                        "to create metadata store."
                    )
                impl = getattr(load_plugin(f'superduper_{backend}'), cls.impl)
                return impl(uri, flavour=flavour)
        raise ValueError(f"No support for uri: {uri}")


class _MetaDataLoader(_Loader):
    impl = 'MetadataStore'
    patterns = {
        r'^mongodb:\/\/': ('mongodb', 'mongodb'),
        r'^mongodb\+srv:\/\/': ('mongodb', 'atlas'),
        r'^mongomock:\/\/': ('mongodb', 'mongomock'),
        r'^sqlite:\/\/': ('sqlalchemy', 'base'),
        r'^postgres:\/\/': ('sqlalchemy', 'base'),
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
        r'^postgres://': ('ibis', 'base'),
        r'^duckdb://': ('ibis', 'base'),
        r'^mssql://': ('ibis', 'base'),
        r'^mysql://': ('ibis', 'base'),
        r'.*\*.csv$': ('ibis', 'pandas'),
    }


class _ArtifactStoreLoader(_Loader):
    impl = 'ArtifactStore'
    patterns = {
        r'^filesystem:\/\/': ('local', 'base'),
        r'^mongodb\+srv:\/\/': ('mongodb', 'atlas'),
        r'^mongodb:\/\/': ('mongodb', 'base'),
    }


def _build_artifact_store(uri, mapping):
    return _ArtifactStoreLoader.create(uri, mapping)


def _build_databackend(uri, mapping):
    return DataBackendProxy(_DataBackendLoader.create(uri, mapping))


def _build_metadata(uri, mapping):
    db = MetaDataStoreProxy(_MetaDataLoader.create(uri, mapping))
    return db


# TODO why public unlike others
def build_compute(cfg):
    """
    Helper function to build compute backend.

    :param cfg: SuperDuper config.
    """

    plugin = load_plugin(cfg.cluster.compute.backend)
    queue_publisher = plugin.QueuePublisher(cfg.cluster.queue.uri)
    return plugin.ComputeBackend(cfg.cluster.compute.uri, queue=queue_publisher)


def build_datalayer(cfg=None, **kwargs) -> Datalayer:
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
    databackend_obj = _build_databackend(cfg.databackend)
    metadata_obj = _build_metadata(cfg.metadata_store or cfg.databackend)
    artifact_store = _build_artifact_store(cfg.artifact_store or cfg.databackend)
    compute = build_compute(cfg)

    datalayer = Datalayer(
        databackend=databackend_obj,
        metadata=metadata_obj,
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
