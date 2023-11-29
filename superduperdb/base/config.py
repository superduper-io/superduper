"""
The classes in this file define the configuration variables for SuperDuperDB,
which means that this file gets imported before alost anything else, and
canot contain any other imports from this project.
"""

import json
import os
import typing as t
from enum import Enum

from .jsonable import Factory, JSONable

_CONFIG_IMMUTABLE = True


class BaseConfigJSONable(JSONable):
    def force_set(self, name, value):
        """
        Forcefully setattr of BaseConfigJSONable instance
        """
        super().__setattr__(name, value)

    def __setattr__(self, name, value):
        if not _CONFIG_IMMUTABLE:
            super().__setattr__(name, value)
            return

        raise AttributeError(
            f'Process attempted to set "{name}" attribute of immutable configuration '
            f'object {self}.'
        )


class Retry(BaseConfigJSONable):
    """
    Describes how to retry using the `tenacity` library

    :param stop_after_attempt: The number of attempts to make
    :param wait_max: The maximum time to wait between attempts
    :param wait_min: The minimum time to wait between attempts
    :param wait_multiplier: The multiplier for the wait time between attempts
    """

    stop_after_attempt: int = 2
    wait_max: float = 10.0
    wait_min: float = 4.0
    wait_multiplier: float = 1.0


class Cluster(BaseConfigJSONable):
    """
    Describes a connection to distributed work via Dask

    :param backfill_batch_size: The number of rows to backfill at a time
                                for vector-search loading
    :param compute: The URI for compute i.e 'local', 'dask+tcp://localhost:8786'
                    "None": Run all jobs in local mode i.e simple function call
                    "local": same as above
                    "dask+thread": Run all jobs on a local threaded dask cluster
                    "dask+tcp://<host>:<port>": Run all jobs on a remote dask cluster

    :param vector_search: The URI for the vector search service
                          "None": Run vector search on local
                          "http://<host>:<port>": Connect a remote vector search service
    :param cdc: The URI for the change data capture service (if "None"
                then no cdc assumed)
                "None": Run cdc on local as a thread.
                "http://<host>:<port>": Connect a remote cdc service
    """

    compute: str = 'local'  # 'dask+tcp://local', 'dask+thread', 'local'
    vector_search: t.Optional[str] = None  # 'http://localhost:8000'  # None
    cdc: t.Optional[str] = None  # 'http://localhost:8001'  # None
    backfill_batch_size: int = 100


class LogLevel(str, Enum):
    """
    Enumerate log severity level
    """

    DEBUG = 'DEBUG'
    INFO = 'INFO'
    SUCCESS = "SUCCESS"
    WARN = 'WARN'
    ERROR = 'ERROR'


class LogType(str, Enum):
    """
    Enumerate the standard logs
    """

    # SYSTEM uses the systems STDOUT and STDERR for printing the logs.
    # DEBUG, INFO, and WARN go to STDOUT.
    # ERROR goes to STDERR.
    SYSTEM = "SYSTEM"

    # LOKI a format that is compatible with the Loki Log aggregation system.
    LOKI = "LOKI"


class Config(BaseConfigJSONable):
    """
    The data class containing all configurable superduperdb values

    :param data_backend: The URI for the data backend
    :param vector_search: The configuration for the vector search {'in_memory', 'lance'}
    :param artifact_store: The URI for the artifact store
    :param metadata_store: The URI for the metadata store
    :param cluster: Settings distributed computing and change data capture
    :param retries: Settings for retrying failed operations

    :param downloads_folder: Settings for downloading files

    :param fold_probability: The probability of validation fold

    :param log_level: The severity level of the logs
    :param logging_type: The type of logging to use

    """

    @property
    def self_hosted_vector_search(self) -> bool:
        return self.data_backend == self.vector_search

    data_backend: str = 'mongodb://superduper:superduper@localhost:27017/test_db'

    vector_search: 'str' = 'in_memory'
    lance_home: str = os.path.join('.superduperdb', 'vector_indices')

    artifact_store: t.Optional[str] = None
    metadata_store: t.Optional[str] = None

    cluster: Cluster = Factory(Cluster)
    retries: Retry = Factory(Retry)

    downloads_folder: t.Optional[str] = None
    fold_probability: float = 0.05

    log_level: LogLevel = LogLevel.DEBUG
    logging_type: LogType = LogType.SYSTEM

    dot_env: t.Optional[str] = None

    class Config(JSONable.Config):
        protected_namespaces = ()

    def __post_init__(self):
        if self.dot_env:
            import dotenv

            dotenv.load_dotenv(self.dot_env)

    @property
    def hybrid_storage(self):
        return self.downloads_folder is not None

    @property
    def comparables(self):
        """
        A dict of `self` excluding some defined attributes.
        """
        _dict = self.dict()
        list(map(_dict.pop, ('cluster', 'retries', 'downloads_folder')))
        return _dict

    def match(self, cfg: dict):
        """
        Match the target cfg dict with `self` comparables dict.
        """
        self_cfg = self.comparables

        self_hash = hash(json.dumps(self_cfg, sort_keys=True))
        cfg_hash = hash(json.dumps(cfg, sort_keys=True))
        return self_hash == cfg_hash

    def force_set(self, name, value):
        """
        Brings immutable behaviour to `CFG` instance.

        CAUTION: Only use it in development mode with caution,
        as this can bring unexpected behaviour.
        """
        parent = self
        names = name.split('.')
        if len(names) > 1:
            name = names[-1]
            for n in names[:-1]:
                parent = getattr(parent, n)
            parent.force_set(name, value)
        else:
            return super().force_set(name, value)
