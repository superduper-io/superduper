import os
import typing as t
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import yaml

from . import config_dicts
from .config import Config, _dataclass_from_dict

File = t.Union[Path, str]

PREFIX = 'SUPERDUPERDB_'
CONFIG_FILE = os.environ.get('SUPERDUPERDB_CONFIG')
USER_CONFIG = Path(CONFIG_FILE).expanduser() if CONFIG_FILE else None
PREFIX = 'SUPERDUPERDB_'
ROOT = Path(__file__).parents[2]


class ConfigError(Exception):
    """An exception raised when there is an error in the configuration."""


@dataclass(frozen=True)
class ConfigSettings:
    """Helper class to read a configuration from a dataclass.

    Reads a dataclass class from a configuration file and environment variables.

    :param cls: The Pydantic class to read.
    :param environ: The environment variables to read from.
    """

    cls: t.Type
    environ: t.Optional[t.Dict] = None

    @cached_property
    def config(self) -> t.Any:
        """Read a configuration using defaults as basis."""
        parent = self.cls().dict()
        env = dict(os.environ if self.environ is None else self.environ)
        env = config_dicts.environ_to_config_dict(PREFIX, parent, env)

        kwargs = {}
        if USER_CONFIG is not None:
            try:
                with open(USER_CONFIG) as f:
                    kwargs = yaml.safe_load(f)
            except FileNotFoundError as e:
                raise ConfigError(f'Could not find config file: {USER_CONFIG}') from e

        kwargs = config_dicts.combine_configs((parent, kwargs, env))

        return _dataclass_from_dict(self.cls, kwargs)


CFG: Config = ConfigSettings(Config).config
