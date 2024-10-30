import os
import typing as t
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import yaml

from . import config_dicts
from .config import _dataclass_from_dict

File = t.Union[Path, str]

HOME = os.environ.get('HOME')

PREFIX = 'SUPERDUPER_'
CONFIG_FILE = os.environ.get('SUPERDUPER_CONFIG')
USER_CONFIG: str = (
    str(Path(CONFIG_FILE).expanduser())
    if CONFIG_FILE
    else (f'{HOME}/.superduper/config.yaml' if HOME else None)
)
PREFIX = 'SUPERDUPER_'
ROOT = Path(__file__).parents[2]


class ConfigError(Exception):
    """
    An exception raised when there is an error in the configuration.

    :param args: *args for `Exception`
    :param kwargs: **kwargs for `Exception`
    """


@dataclass(frozen=True)
class ConfigSettings:
    """Helper class to read a configuration from a dataclass.

    Reads a dataclass class from a configuration file and environment variables.

    :param cls: The Pydantic class to read.
    :param environ: The environment variables to read from.
    :param base: The base field of a loaded config file to use
                 (e.g. "cluster" loads from r["cluster"])
    """

    cls: t.Type
    environ: t.Optional[t.Dict] = None
    base: t.Optional[str] = None

    @cached_property
    def config(self) -> t.Any:
        """Read a configuration using defaults as basis."""
        parent = self.cls().dict()
        env = dict(os.environ if self.environ is None else self.environ)
        prefix = PREFIX
        if self.base:
            prefix = PREFIX + self.base.upper() + '_'

        env = config_dicts.environ_to_config_dict(prefix, parent, env)

        kwargs = {}
        if USER_CONFIG is not None:
            try:
                with open(USER_CONFIG) as f:
                    kwargs = yaml.safe_load(f)
            except FileNotFoundError as e:
                if USER_CONFIG != f'{HOME}/.superduper/config.yaml':
                    raise ConfigError(
                        f'Could not find config file: {USER_CONFIG}'
                    ) from e
            if self.base:
                kwargs = kwargs.get(self.base, {})

        kwargs = config_dicts.combine_configs((parent, kwargs, env))

        return _dataclass_from_dict(self.cls, kwargs)
