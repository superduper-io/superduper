import os
import typing as t
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

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
    cls: t.Type
    default_files: t.Union[t.Sequence[Path], str]
    prefix: str
    environ: t.Optional[t.Dict] = None

    @cached_property
    def config(self) -> t.Any:
        """Read a Pydantic class"""
        environ = dict(os.environ if self.environ is None else self.environ)

        files = environ.pop(self.prefix + FILES_NAME, self.default_files)
        if isinstance(files, str):
            files = files.split(FILE_SEP)

        data = config_dicts.read_all(files)
        parent = self.cls().dict()
        environ_dict = config_dicts.environ_to_config_dict(self.prefix, parent, environ)
        return self.cls(**config_dicts.combine((*data, environ_dict)))


def build_config() -> Config:
    """
    Build the config object from the environment variables and config files.
    """
    CONFIG = ConfigSettings(Config, _ALL_CONFIGS, PREFIX)
    return CONFIG.config


CFG = build_config()
