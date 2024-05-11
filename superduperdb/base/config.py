"""Configuration variables for SuperDuperDB.

The classes in this file define the configuration variables for SuperDuperDB,
which means that this file gets imported before alost anything else, and
canot contain any other imports from this project.
"""

import dataclasses as dc
import json
import os
import typing as t
from enum import Enum


def _dataclass_from_dict(data_class: t.Any, data: dict):
    field_types = {f.name: f.type for f in data_class.__dataclass_fields__.values()}
    params = {}
    for f in data:
        if (
            f in field_types
            and hasattr(field_types[f], '__dataclass_fields__')
            and not isinstance(data[f], field_types[f])
        ):
            params[f] = _dataclass_from_dict(field_types[f], data[f])
        else:
            params[f] = data[f]
    return data_class(**params)


@dc.dataclass
class BaseConfig:
    """A base class for configuration dataclasses.

    This class allows for easy updating of configuration dataclasses
    with a dictionary of parameters.
    """

    def __call__(self, **kwargs):
        """Update the configuration with the given parameters."""
        parameters = self.dict()
        for k, v in kwargs.items():
            if '__' in k:
                parts = k.split('__')
                parent = parts[0]
                child = '__'.join(parts[1:])
                parameters[parent] = getattr(self, parent)(**{child: v})
            else:
                parameters[k] = v
        return _dataclass_from_dict(type(self), parameters)

    def dict(self):
        """Return the configuration as a dictionary."""
        return dc.asdict(self)


@dc.dataclass
class Retry(BaseConfig):
    """Describes how to retry using the `tenacity` library.

    :param stop_after_attempt: The number of attempts to make
    :param wait_max: The maximum time to wait between attempts
    :param wait_min: The minimum time to wait between attempts
    :param wait_multiplier: The multiplier for the wait time between attempts
    """

    stop_after_attempt: int = 2
    wait_max: float = 10.0
    wait_min: float = 4.0
    wait_multiplier: float = 1.0


@dc.dataclass
class CDCStrategy:
    """Base CDC strategy dataclass.

    :param type: The type of CDC strategy
    """

    type: str


@dc.dataclass
class PollingStrategy(CDCStrategy):
    """Describes a polling strategy for change data capture.

    :param auto_increment_field: The field to use for auto-incrementing
    :param frequency: The frequency to poll for changes
    :param type: The type of CDC strategy
    """

    auto_increment_field: t.Optional[str] = None
    frequency: float = 3600
    type: 'str' = 'incremental'


@dc.dataclass
class LogBasedStrategy(CDCStrategy):
    """Describes a log-based strategy for change data capture.

    :param resume_token: The resume token to use for log-based CDC
    :param type: The type of CDC strategy
    """

    resume_token: t.Optional[t.Dict[str, str]] = None
    type: str = 'logbased'


@dc.dataclass
class CDCConfig(BaseConfig):
    """Describes the configuration for change data capture.

    :param uri: The URI for the CDC service
    :param strategy: The strategy to use for CDC
    """

    uri: t.Optional[str] = None  # None implies local mode
    strategy: t.Optional[t.Union[PollingStrategy, LogBasedStrategy]] = None


@dc.dataclass
class VectorSearch(BaseConfig):
    """Describes the configuration for vector search.

    :param uri: The URI for the vector search service
    :param type: The type of vector search service
    :param backfill_batch_size: The size of the backfill batch
    """

    uri: t.Optional[str] = None  # None implies local mode
    type: str = 'in_memory'  # in_memory|lance
    backfill_batch_size: int = 100


@dc.dataclass
class Rest(BaseConfig):
    """Describes the configuration for the REST service.

    :param uri: The URI for the REST service
    """

    uri: t.Optional[str] = None


@dc.dataclass
class Compute(BaseConfig):
    """Describes the configuration for distributed computing.

    :param uri: The URI for the compute service
    :param compute_kwargs: The keyword arguments to pass to the compute service
    """

    uri: t.Optional[str] = None  # None implies local mode
    compute_kwargs: t.Dict = dc.field(default_factory=dict)


@dc.dataclass
class Cluster(BaseConfig):
    """Describes a connection to distributed work via Ray.

    :param compute: The URI for compute
                    - None: run all jobs in local mode i.e. simple function call
                    - "ray://<host>:<port>": Run all jobs on a remote ray cluster
    :param vector_search: The URI for the vector search service
                          None: Run vector search on local
                          "http://<host>:<port>": Connect a remote vector search service
    :param rest: The URI for the REST service
    :param cdc: The URI for the change data capture service (if "None"
                then no cdc assumed)
                None: Run cdc on local as a thread.
                "http://<host>:<port>": Connect a remote cdc service
    """

    compute: Compute = dc.field(default_factory=Compute)
    vector_search: VectorSearch = dc.field(default_factory=VectorSearch)
    rest: Rest = dc.field(default_factory=Rest)
    cdc: CDCConfig = dc.field(default_factory=CDCConfig)


class LogLevel(str, Enum):
    """Enumerate log severity level."""

    DEBUG = 'DEBUG'
    INFO = 'INFO'
    SUCCESS = "SUCCESS"
    WARN = 'WARN'
    ERROR = 'ERROR'


class LogType(str, Enum):
    """Enumerate the standard logs."""

    # SYSTEM uses the systems STDOUT and STDERR for printing the logs.
    # DEBUG, INFO, and WARN go to STDOUT.
    # ERROR goes to STDERR.
    SYSTEM = "SYSTEM"

    # LOKI a format that is compatible with the Loki Log aggregation system.
    LOKI = "LOKI"


class BytesEncoding(str, Enum):
    """Enumerate the encoding of bytes in the data backend."""

    BYTES = 'Bytes'
    BASE64 = 'Str'


@dc.dataclass
class Downloads(BaseConfig):
    """Describes the configuration for downloading files.

    :param folder: The folder to download files to
    :param n_workers: The number of workers to use for downloading
    :param headers: The headers to use for downloading
    :param timeout: The timeout for downloading
    """

    folder: t.Optional[str] = None
    n_workers: int = 0
    headers: t.Dict = dc.field(default_factory=lambda: {'User-Agent': 'me'})
    timeout: t.Optional[int] = None


@dc.dataclass
class Config(BaseConfig):
    """The data class containing all configurable superduperdb values.

    :param envs: The envs datas
    :param data_backend: The URI for the data backend
    :param lance_home: The home directory for the Lance vector indices,
                       Default: .superduperdb/vector_indices
    :param artifact_store: The URI for the artifact store
    :param metadata_store: The URI for the metadata store
    :param cluster: Settings distributed computing and change data capture
    :param retries: Settings for retrying failed operations
    :param downloads: Settings for downloading files
    :param fold_probability: The probability of validation fold
    :param log_level: The severity level of the logs
    :param logging_type: The type of logging to use
    :param bytes_encoding: The encoding of bytes in the data backend

    """

    envs: dc.InitVar[t.Optional[t.Dict[str, str]]] = None

    data_backend: str = 'mongodb://localhost:27017/test_db'
    lance_home: str = os.path.join('.superduperdb', 'vector_indices')

    artifact_store: t.Optional[str] = None
    metadata_store: t.Optional[str] = None

    cluster: Cluster = dc.field(default_factory=Cluster)
    retries: Retry = dc.field(default_factory=Retry)
    downloads: Downloads = dc.field(default_factory=Downloads)

    fold_probability: float = 0.05

    log_level: LogLevel = LogLevel.INFO
    logging_type: LogType = LogType.SYSTEM

    bytes_encoding: BytesEncoding = BytesEncoding.BYTES

    def __post_init__(self, envs):
        if envs is not None:
            for k, v in envs.items():
                os.environ[k.upper()] = v

    @property
    def hybrid_storage(self):
        """Whether to use hybrid storage."""
        return self.downloads.folder is not None

    @property
    def comparables(self):
        """A dict of `self` excluding some defined attributes."""
        _dict = dc.asdict(self)
        list(map(_dict.pop, ('cluster', 'retries', 'downloads')))
        return _dict

    def match(self, cfg: t.Dict):
        """Match the target cfg dict with `self` comparables dict.

        :param cfg: The target configuration dictionary.
        """
        self_cfg = self.comparables
        self_hash = hash(json.dumps(self_cfg, sort_keys=True))
        cfg_hash = hash(json.dumps(cfg, sort_keys=True))
        return self_hash == cfg_hash

    def diff(self, cfg: t.Dict):
        """Return the difference between `self` and the target cfg dict.

        :param cfg: The target configuration dictionary.
        """
        return _diff(self.dict(), cfg)

    def to_yaml(self):
        """Return the configuration as a YAML string."""
        import yaml

        def enum_representer(dumper, data):
            return dumper.represent_scalar('tag:yaml.org,2002:str', str(data.value))

        yaml.SafeDumper.add_representer(BytesEncoding, enum_representer)
        yaml.SafeDumper.add_representer(LogLevel, enum_representer)
        yaml.SafeDumper.add_representer(LogType, enum_representer)

        return yaml.dump(self.dict(), Dumper=yaml.SafeDumper)


def _diff(r1, r2):
    """Return the difference between two dictionaries.

    >>> _diff({'a': 1, 'b': 2}, {'a': 2, 'b': 2})
    {'a': (1, 2)}
    >>> _diff({'a': {'c': 3}, 'b': 2}, {'a': 2, 'b': 2})
    {'a': ({'c': 3}, 2)}
    """
    d = _diff_impl(r1, r2)
    out = {}
    for path, left, right in d:
        out['.'.join(path)] = (left, right)
    return out


def _diff_impl(r1, r2):
    if not isinstance(r1, dict) or not isinstance(r2, dict):
        if r1 == r2:
            return []
        return [([], r1, r2)]
    out = []
    for k in list(r1.keys()) + list(r2.keys()):
        if k not in r1:
            out.append(([k], None, r2[k]))
            continue
        if k not in r2:
            out.append(([k], r1[k], None))
            continue
        out.extend([([k, *x[0]], x[1], x[2]) for x in _diff_impl(r1[k], r2[k])])
    return out
