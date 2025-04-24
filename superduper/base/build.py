import re
import typing as t
from typing import Callable

import deprecation
from prettytable import PrettyTable

import superduper as s
from superduper import logging
from superduper.backends.base.compute import ComputeBackend
from superduper.backends.base.data_backend import DataBackendProxy
from superduper.backends.local.cache import LocalCache
from superduper.base.artifacts import FileSystemArtifactStore
from superduper.base.config import Config
from superduper.base.datalayer import Datalayer
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
    }


class Builder:
    r"""
    Builder for creating a Datalayer instance with a fluent interface.

    This class provides a functional-style parameterization pattern to configure
    various components of a Datalayer before building the final instance.

    :Example:

    . code-block:: python

        datalayer = Builder() \\
            .with_compute(engine) \\
            .with_data_backend("path/to/data") \\
            .with_local_cache() \\
            .build()
    """

    def __init__(self):
        # Core components
        self._data_backend = None
        self._artifact_store = None
        self._cache = None
        self._compute_callback = None

    def with_data_backend(self, uri: str) -> 'Builder':
        """
        Configure the data backend using the provided URI.

        :param uri: Location identifier for the data backend
        :return: Self for method chaining
        """
        self._data_backend = DataBackendProxy(_DataBackendLoader).create(uri)
        return self

    def with_artifact_store(self, uri: str) -> 'Builder':
        """
        Set the artifact store using a filesystem path.

        :param uri: Path to the artifact store location
        :return: Self for method chaining
        """
        self._artifact_store = FileSystemArtifactStore(uri)
        return self

    def with_local_cache(self) -> 'Builder':
        """
        Use a local in-memory cache implementation.

        :return: Self for method chaining
        """
        self._cache = LocalCache()
        return self

    def with_redis_cache(self, uri: str) -> 'Builder':
        """
        Configure Redis as the caching backend.

        :param uri: Redis connection URI
        :return: Self for method chaining
        """
        self._cache = load_plugin('redis').Cache(uri=uri)
        return self

    def with_compute(
        self, callback: 'Callable[[Datalayer], ComputeBackend]'
    ) -> 'Builder':
        """
        Set the compute backend using a callback function.

        :param callback: A function that receives the datalayer being built
                        and returns a configured compute backend
        :return: Self for method chaining

        :Example:

        . code-block:: python

            def setup_compute(datalayer: 'Datalayer') -> 'ComputeBackend':
                return LocalComputeBackend(datalayer)

            datalayer = Builder().with_compute(setup_compute).build()
        """
        self._compute_callback = callback
        return self

    def build(self) -> 'Datalayer':
        """
        Build and initialize the configured Datalayer instance.

        :return: A fully initialized Datalayer instance

        . note::
            This method first creates the Datalayer with the configured components,
            then initializes it with the compute backend.
        """
        # Create datalayer with configured components
        datalayer = Datalayer(
            databackend=self._data_backend,
            artifact_store=self._artifact_store,
            cache=self._cache,
        )

        # Set up compute backend using the callback
        cluster = self._compute_callback(datalayer)

        # Initialize datalayer with the compute backend
        datalayer.initialize(cluster)

        logging.info("New Datalayer has been created")

        return datalayer


@deprecation.deprecated(details="Use Builder() instead.")
def build_datalayer(
    cfg=None, compute: ComputeBackend | None = None, **kwargs
) -> 'Datalayer':
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

    # Lazy imports for builder functions to avoid circular dependencies
    databackend_obj = DataBackendProxy(_DataBackendLoader).create(cfg.data_backend)
    artifact_store = FileSystemArtifactStore(CFG.artifact_store)

    # Create the cache
    cache = None
    if CFG.cache and CFG.cache.startswith('redis'):
        cache = load_plugin('redis').Cache(uri=CFG.cache)
    elif CFG.cache:
        assert CFG.cache == 'in-process'
        cache = LocalCache()
    # --------------

    # Create datalayer without the cluster
    datalayer = Datalayer(
        databackend=databackend_obj,
        artifact_store=artifact_store,
        cache=cache,
    )

    # Create the cluster
    backend = getattr(load_plugin(cfg.cluster_engine), 'Cluster')
    cluster = backend.build(cfg, compute=compute, db=datalayer)

    # Initialize the data layer with the cluster
    datalayer.initialize(cluster)

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
    key_values = [
        ('Data Backend', anonymize_url(cfg.data_backend)),
        ('Artifact Store', anonymize_url(cfg.artifact_store)),
        ('Cache', anonymize_url(cfg.cache)),
    ]
    for key, value in key_values:
        if value:
            table.add_row([key, value])

    logging.info(f"Configuration: \n {table}")
