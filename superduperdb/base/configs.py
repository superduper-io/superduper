import os
import typing as t
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import yaml

from . import config_dicts
from .config import Config

File = t.Union[Path, str]

# The top-level directory of the project
ROOT = Path(__file__).parents[2]

# The default prefix used for config environment variables
PREFIX = 'SUPERDUPERDB_'

# The name of the environment variable used to read the config files.
# This value needs to be read before all the other config values are.
FILES_NAME = 'CONFIG_FILES'

# The base name of the configs file
CONFIG_FILE = 'configs.json'

_LOCAL_CONFIG = Path(CONFIG_FILE)
_PROJECT_CONFIG = ROOT / CONFIG_FILE
_USER_CONFIG = Path(f'~/.superduperdb/{CONFIG_FILE}').expanduser()

_ALL_CONFIGS = _PROJECT_CONFIG, _LOCAL_CONFIG, _USER_CONFIG

FILE_SEP = ','


@dataclass(frozen=True)
class ConfigSettings:
    """
    A class that reads a Pydantic class from config files and environment variables.

    :param cls: The Pydantic class to read.
    :param default_files: The default config files to read.
    :param prefix: The prefix to use for environment variables.
    :param environ: The environment variables to read from.
    """

    cls: t.Type
    default_files: t.Union[t.Sequence[Path], str]
    prefix: str
    environ: t.Optional[t.Dict] = None
    base_config: t.Optional[Config] = None

    @cached_property
    def config(self) -> t.Any:
        """Read a Pydantic class"""

        if self.base_config:
            parent = self.base_config.dict()
        else:
            parent = self.cls().dict()

        env = dict(os.environ if self.environ is None else self.environ)
        env = config_dicts.environ_to_config_dict('SUPERDUPERDB_', parent, env)

        config_path = '.superduperdb/config.yaml'
        if os.path.exists(config_path):
            with open(config_path) as f:
                kwargs = yaml.safe_load(f)
        else:
            kwargs = {}

        kwargs = config_dicts.combine_configs((parent, kwargs, env))
        return self.cls(**kwargs)


def build_config(cfg: t.Optional[Config] = None) -> Config:
    """
    Build the config object from the environment variables and config files.
    """
    CONFIG = ConfigSettings(Config, _ALL_CONFIGS, PREFIX, base_config=cfg)
    return CONFIG.config


CFG = build_config()
