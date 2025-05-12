"""Configuration variables for superduper.io.

The classes in this file define the configuration variables for superduper.io,
hich means that this file gets imported before alost anything else, and
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
            and hasattr(field_types[f], "__dataclass_fields__")
            and not isinstance(data[f], field_types[f])
        ):
            params[f] = _dataclass_from_dict(field_types[f], data[f])
        else:
            params[f] = data[f]
    valid_params = {k: v for k, v in params.items() if k in field_types}
    return data_class(**valid_params)


@dc.dataclass
class BaseConfig:
    """A base class for configuration dataclasses.

    This class allows for easy updating of configuration dataclasses
    with a dictionary of parameters.
    """

    def __call__(self, **kwargs):
        """Update the configuration with the given parameters."""
        parameters = dc.asdict(self)
        for k, v in kwargs.items():
            if "__" in k:
                parts = k.split("__")
                parent = parts[0]
                child = "__".join(parts[1:])
                parameters[parent] = getattr(self, parent)(**{child: v})
            else:
                parameters[k] = v
        out = _dataclass_from_dict(type(self), parameters)
        if hasattr(self, 'cluster'):
            out.cluster = self.cluster
        return out

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


class LogLevel(str, Enum):
    """Enumerate log severity level # noqa."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARN = "WARN"
    ERROR = "ERROR"


class LogType(str, Enum):
    """Enumerate the standard logs # noqa."""

    # SYSTEM uses the systems STDOUT and STDERR for printing the logs.
    # DEBUG, INFO, and WARN go to STDOUT.
    # ERROR goes to STDERR.
    SYSTEM = "SYSTEM"

    # LOKI a format that is compatible with the Loki Log aggregation system.
    LOKI = "LOKI"


class BytesEncoding(str, Enum):
    """Enumerate the encoding of bytes in the data backend # noqa."""

    BYTES = "bytes"
    BASE64 = "str"


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
    headers: t.Dict = dc.field(default_factory=lambda: {"User-Agent": "me"})
    timeout: t.Optional[int] = None


@dc.dataclass
class DataTypePresets(BaseConfig):
    """Paths of default types of data.

    Overrides DataBackend.datatype_presets.

    :param vector: BaseDataType to encode vectors.
    """

    vector: str | None = None


@dc.dataclass
class Config(BaseConfig):
    """The data class containing all configurable superduper values.

    :param envs: The envs datas
    :param data_backend: The URI for the data backend
    :param secrets_volume: The secrets volume mount for secrets env vars.
    :param artifact_store: The URI for the artifact store
    :param metadata_store: The URI for the metadata store
    :param cache: A URI for an in-memory cache
    :param vector_search_engine: The engine to use for vector search
    :param cluster_engine: The engine to use for operating a distributed cluster
    :param retries: Settings for retrying failed operations
    :param downloads: Settings for downloading files
    :param log_level: The severity level of the logs
    :param logging_type: The type of logging to use
    :param log_hostname: Whether to include the hostname in the logs
    :param force_apply: Whether to force apply the configuration
    :param datatype_presets: Presets to be applied for default types of data
    :param json_native: Whether the databackend supports json natively or not.
    :param log_colorize: Whether to colorize the logs
    :param bytes_encoding: (Deprecated)
    :param output_prefix: The prefix for the output table and output field key
    :param vector_search_kwargs: The keyword arguments to pass to the vector search
    """

    envs: dc.InitVar[t.Optional[t.Dict[str, str]]] = None

    secrets_volume: str = os.path.join(".superduper", "/session/secrets")
    data_backend: str = "mongodb://localhost:27017/test_db"

    artifact_store: str = 'filesystem://./artifact_store'
    metadata_store: str = ''

    cache: str | None = None
    vector_search_engine: str = 'local'
    cluster_engine: str = 'local'

    retries: Retry = dc.field(default_factory=Retry)
    downloads: Downloads = dc.field(default_factory=Downloads)

    log_level: LogLevel = LogLevel.INFO
    logging_type: LogType = LogType.SYSTEM
    log_colorize: bool = True
    log_hostname: bool = True
    bytes_encoding: str = 'bytes'

    force_apply: bool = False

    datatype_presets: DataTypePresets = dc.field(default_factory=DataTypePresets)

    json_native: bool = True
    output_prefix: str = "_outputs__"
    vector_search_kwargs: t.Dict = dc.field(default_factory=dict)

    def __post_init__(self, envs):
        if envs is not None:
            for k, v in envs.items():
                os.environ[k.upper()] = v
        self.secrets_volume = os.path.expanduser(self.secrets_volume)

    @property
    def comparables(self):
        """A dict of `self` excluding some defined attributes."""
        _dict = dc.asdict(self)
        if hasattr(self, 'cluster'):
            _dict.update({'cluster': dc.asdict(self.cluster)})
        list(map(_dict.pop, ("cluster", "retries", "downloads")))
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
            return dumper.represent_scalar("tag:yaml.org,2002:str", str(data.value))

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
        out[".".join(path)] = (left, right)
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
