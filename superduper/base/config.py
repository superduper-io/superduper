"""Configuration variables for superduper.io.

The classes in this file define the configuration variables for superduper.io,
which means that this file gets imported before almost anything else, and
cannot contain any other imports from this project.
"""

import dataclasses as dc
import json
import os
import typing as t
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union


def _dataclass_from_dict(data_class: Type[Any], data: Dict[str, Any]) -> Any:
    """Convert a dictionary to a dataclass instance.

    Args:
        data_class: The dataclass type to instantiate
        data: Dictionary containing the data to populate the dataclass

    Returns:
        An instance of the dataclass populated with values from the dictionary
    """
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

    def __call__(self, **kwargs: Any) -> 'BaseConfig':
        """Update the configuration with the given parameters.

        Args:
            **kwargs: The parameters to update

        Returns:
            A new instance of the configuration with updated parameters
        """
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

    def dict(self) -> Dict[str, Any]:
        """Return the configuration as a dictionary.

        Returns:
            A dictionary representation of the configuration
        """
        return dc.asdict(self)


@dc.dataclass
class Retry(BaseConfig):
    """Describes how to retry using the `tenacity` library.

    Args:
        stop_after_attempt: The number of attempts to make
        wait_max: The maximum time to wait between attempts (seconds)
        wait_min: The minimum time to wait between attempts (seconds)
        wait_multiplier: The multiplier for the wait time between attempts
    """

    stop_after_attempt: int = 2
    wait_max: float = 10.0
    wait_min: float = 4.0
    wait_multiplier: float = 1.0


class LogLevel(str, Enum):
    """Enumerate log severity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARN = "WARN"
    ERROR = "ERROR"


class LogType(str, Enum):
    """Enumerate the standard log output types."""

    # SYSTEM uses the systems STDOUT and STDERR for printing the logs.
    # DEBUG, INFO, and WARN go to STDOUT.
    # ERROR goes to STDERR.
    SYSTEM = "SYSTEM"

    # LOKI uses a format that is compatible with the Loki Log aggregation system.
    LOKI = "LOKI"


class BytesEncoding(str, Enum):
    """Enumerate the encoding of bytes in the data backend."""

    BYTES = "bytes"  # Raw bytes encoding
    BASE64 = "str"   # Base64 string encoding


@dc.dataclass
class Downloads(BaseConfig):
    """Describes the configuration for downloading files.

    Args:
        folder: The folder to download files to
        n_workers: The number of workers to use for downloading
        headers: The HTTP headers to use for downloading
        timeout: The timeout for downloads in seconds
    """

    folder: Optional[str] = None
    n_workers: int = 0
    headers: Dict[str, str] = dc.field(default_factory=lambda: {"User-Agent": "me"})
    timeout: Optional[int] = None


@dc.dataclass
class DataTypePresets(BaseConfig):
    """Paths of default types of data.

    Overrides DataBackend.datatype_presets.

    Args:
        vector: BaseDataType to encode vectors.
    """

    vector: Optional[str] = None


@dc.dataclass
class Config(BaseConfig):
    """The data class containing all configurable superduper values.

    Args:
        envs: Environment variables to set
        secrets_volume: The secrets volume mount for secrets env vars
        data_backend: The URI for the data backend
        artifact_store: The URI for the artifact store
        metadata_store: The URI for the metadata store
        cache: A URI for an in-memory cache
        vector_search_engine: The engine to use for vector search
        cluster_engine: The engine to use for operating a distributed cluster
        retries: Settings for retrying failed operations
        downloads: Settings for downloading files
        log_level: The severity level of the logs
        logging_type: The type of logging to use
        log_colorize: Whether to colorize the logs
        bytes_encoding: Encoding for bytes data (deprecated)
        force_apply: Whether to force apply the configuration
        datatype_presets: Presets to be applied for default types of data
        json_native: Whether the databackend supports JSON natively
        output_prefix: The prefix for the output table and output field key
        vector_search_kwargs: The keyword arguments to pass to the vector search
    """

    envs: dc.InitVar[Optional[Dict[str, str]]] = None

    secrets_volume: str = os.path.join(".superduper", "session/secrets")
    data_backend: str = "mongodb://localhost:27017/test_db"

    # TODO drop the "filesystem://" prefix
    artifact_store: str = 'filesystem://./artifact_store'
    metadata_store: Optional[str] = None
    cache: str = 'in-process'
    vector_search_engine: str = 'local'
    cluster_engine: str = 'local'

    retries: Retry = dc.field(default_factory=Retry)
    downloads: Downloads = dc.field(default_factory=Downloads)

    log_level: LogLevel = LogLevel.INFO
    logging_type: LogType = LogType.SYSTEM
    log_colorize: bool = True
    bytes_encoding: str = 'bytes'

    force_apply: bool = False

    datatype_presets: DataTypePresets = dc.field(default_factory=DataTypePresets)

    json_native: bool = True
    output_prefix: str = "_outputs__"
    vector_search_kwargs: Dict[str, Any] = dc.field(default_factory=dict)

    def __post_init__(self, envs: Optional[Dict[str, str]]) -> None:
        """Initialize the configuration after __init__.

        Sets environment variables if provided and expands the secrets volume path.

        Args:
            envs: Dictionary of environment variables to set
        """
        if envs is not None:
            for k, v in envs.items():
                os.environ[k.upper()] = v
        self.secrets_volume = os.path.expanduser(self.secrets_volume)

    @property
    def comparables(self) -> Dict[str, Any]:
        """A dict of `self` excluding some defined attributes.

        Returns:
            Dictionary of configuration values excluding cluster, retries, and downloads
        """
        _dict = dc.asdict(self)
        if hasattr(self, 'cluster'):
            _dict.update({'cluster': dc.asdict(self.cluster)})
        for key in ("cluster", "retries", "downloads"):
            _dict.pop(key, None)
        return _dict

    def match(self, cfg: Dict[str, Any]) -> bool:
        """Match the target cfg dict with `self` comparables dict.

        Args:
            cfg: The target configuration dictionary

        Returns:
            True if configurations match, False otherwise
        """
        self_cfg = self.comparables
        self_hash = hash(json.dumps(self_cfg, sort_keys=True))
        cfg_hash = hash(json.dumps(cfg, sort_keys=True))
        return self_hash == cfg_hash

    def diff(self, cfg: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        """Return the difference between `self` and the target cfg dict.

        Args:
            cfg: The target configuration dictionary

        Returns:
            Dictionary of differences with keys as paths and values as tuples of (self_value, cfg_value)
        """
        return _diff(self.dict(), cfg)

    def to_yaml(self) -> str:
        """Return the configuration as a YAML string.

        Returns:
            YAML representation of the configuration
        """
        import yaml

        def enum_representer(dumper: yaml.SafeDumper, data: Enum) -> yaml.ScalarNode:
            """Custom representer for Enum values in YAML."""
            return dumper.represent_scalar("tag:yaml.org,2002:str", str(data.value))

        yaml.SafeDumper.add_representer(BytesEncoding, enum_representer)
        yaml.SafeDumper.add_representer(LogLevel, enum_representer)
        yaml.SafeDumper.add_representer(LogType, enum_representer)

        return yaml.dump(self.dict(), Dumper=yaml.SafeDumper)


def _diff(r1: Dict[str, Any], r2: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    """Return the difference between two dictionaries.

    Args:
        r1: First dictionary
        r2: Second dictionary

    Returns:
        Dictionary with keys as dot-notated paths and values as tuples of (r1_value, r2_value)

    Examples:
        >>> _diff({'a': 1, 'b': 2}, {'a': 2, 'b': 2})
        {'a': (1, 2)}
        >>> _diff({'a': {'c': 3}, 'b': 2}, {'a': 2, 'b': 2})
        {'a': ({'c': 3}, 2)}
    """
    d = _diff_impl(r1, r2)
    out: Dict[str, Tuple[Any, Any]] = {}
    for path, left, right in d:
        out[".".join(path)] = (left, right)
    return out


def _diff_impl(r1: Any, r2: Any) -> List[Tuple[List[str], Any, Any]]:
    """Implementation for _diff that returns path parts instead of dot notation.

    Args:
        r1: First value (dictionary or primitive)
        r2: Second value (dictionary or primitive)

    Returns:
        List of tuples containing (path_parts, r1_value, r2_value)
    """
    if not isinstance(r1, dict) or not isinstance(r2, dict):
        if r1 == r2:
            return []
        return [([], r1, r2)]

    out: List[Tuple[List[str], Any, Any]] = []
    for k in set(list(r1.keys()) + list(r2.keys())):
        if k not in r1:
            out.append(([k], None, r2[k]))
            continue
        if k not in r2:
            out.append(([k], r1[k], None))
            continue
        out.extend([([k, *x[0]], x[1], x[2]) for x in _diff_impl(r1[k], r2[k])])
    return out