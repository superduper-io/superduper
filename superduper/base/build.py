import re
import typing as t

from prettytable import PrettyTable

import superduper as s
from superduper import CFG, logging
from superduper.backends.base.compute import ComputeBackend
from superduper.backends.base.data_backend import DataBackendProxy
from superduper.base.artifacts import (
    FileSystemArtifactStore,
)
from superduper.base.config import Config
from superduper.base.datalayer import Datalayer
from superduper.base.metadata import MetaDataStore
from superduper.misc.anonymize import anonymize_url
from superduper.misc.importing import load_plugin


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
        """Helper method to create backend."""
        plugin, flavour = cls.match(uri)
        if cls.not_supported and (plugin, flavour) in cls.not_supported:
            raise ValueError(
                f"{plugin} with flavour {flavour} not supported " "to create store."
            )
        plugin = load_plugin(plugin)
        impl = getattr(plugin, cls.impl)
        return impl(uri, flavour=flavour, plugin=plugin)


class _DataBackendLoader(_Loader):
    impl = 'DataBackend'
    patterns = {
        r'^mongodb:\/\/': ('mongodb', 'mongodb'),
        r'^mongodb\+srv:\/\/': ('mongodb', 'atlas'),
        r'^mongomock:\/\/': ('mongodb', 'mongomock'),
        r'^sqlite://': ('sql', 'base'),
        r'^postgresql://': ('sql', 'base'),
        r'^snowflake:\/\/': ('snowflake', 'base'),
        r'^duckdb://': ('sql', 'base'),
        r'^mssql://': ('sql', 'base'),
        r'^mysql://': ('sql', 'base'),
        r'^redis://': ('redis', 'base'),
        r'^inmemory://': ('inmemory', 'base'),
    }


def _build_artifact_store():
    return FileSystemArtifactStore(CFG.artifact_store)


def _build_databackend(uri):
    return DataBackendProxy(_DataBackendLoader.create(uri))


def build_datalayer(
    cfg=None, compute: ComputeBackend | None = None, **kwargs
) -> Datalayer:
    """
    Build a Datalayer object as per ``db = superduper(db)`` from configuration.

    :param cfg: Configuration to use. If None, use ``superduper.CFG``.
    :param kwargs: keyword arguments to be adopted by the `CFG`
    """
    # Configuration
    # ------------------------------
    # Use the provided configuration or fall back to the default configuration.
    if s.CFG.cluster_engine != 'local':
        plugin = load_plugin(s.CFG.cluster_engine)
        CFG = getattr(plugin, 'CFG')
    else:
        CFG = s.CFG

    cfg = (cfg or CFG)(**kwargs)

    cfg = t.cast(Config, cfg)
    databackend_obj = _build_databackend(cfg.data_backend)

    artifact_store = _build_artifact_store()

    backend = getattr(load_plugin(cfg.cluster_engine), 'Cluster')

    cluster = backend.build(cfg, compute=compute)

    if cfg.metadata_store:
        metadata = _build_databackend(cfg.metadata_store)
        metadata = Datalayer(
            databackend=metadata,
            cluster=cluster,
            artifact_store=artifact_store,
            metadata=None,
        )
    else:
        metadata = None

    datalayer = Datalayer(
        databackend=databackend_obj,
        artifact_store=artifact_store,
        cluster=cluster,
        metadata=metadata,
    )
    # Keep the real configuration in the datalayer object.
    datalayer.cfg = cfg

    if kwargs.get('initialize_cluster', True):
        assert datalayer.cluster is not None
        datalayer.cluster.initialize()

    show_configuration(cfg)
    return datalayer


def show_configuration(cfg):
    """Show the configuration.

    Only show the important configuration values and anonymize the URLs.

    :param cfg: The configuration object.
    """
    table = PrettyTable()
    table.field_names = ["Configuration", "Value"]
    key_values = [
        ('DataBackend', anonymize_url(cfg.data_backend)),
        ('ArtifactStore', anonymize_url(cfg.artifact_store)),
        ('Metadata', anonymize_url(cfg.metadata_store)),
    ]
    for key, value in key_values:
        if value:
            table.add_row([key, value])

    logging.info(f"Configuration: \n {table}")
