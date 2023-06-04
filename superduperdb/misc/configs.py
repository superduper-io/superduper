from . import config
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from superduperdb.misc import dicts
from typing import Any, Dict, Optional, Sequence, Type, Union
import os

Self = Any
File = Union[Path, str]

ROOT = Path(__file__).parents[1]
PREFIX = 'SUPERDUPERDB_'
FILES_NAME = 'CONFIG_FILES'

CONFIG_FILE = 'configs.json'

LOCAL_CONFIG = Path(CONFIG_FILE)
PROJECT_CONFIG = ROOT / CONFIG_FILE
USER_CONFIG = Path(f'~/.superduperdb/{CONFIG_FILE}').expanduser()

ALL_CONFIGS = PROJECT_CONFIG, LOCAL_CONFIG, USER_CONFIG

FILE_SEP = ','


@dataclass(frozen=True)
class ConfigSettings:
    cls: Type
    default_files: Union[Sequence[Path], str]
    prefix: str
    environ: Optional[Dict] = None

    @cached_property
    def config(self) -> Any:
        """Read a Pydantic class"""
        environ = dict(os.environ if self.environ is None else self.environ)

        files = environ.pop(self.prefix + FILES_NAME, self.default_files)
        if isinstance(files, str):
            files = files.split(FILE_SEP)

        data = dicts.read_all(files)
        parent = self.cls().dict()
        environ_dict = dicts.environ_to_config_dict(self.prefix, parent, environ)
        return self.cls(**dicts.combine((*data, environ_dict)))


CONFIG = ConfigSettings(config.Config, ALL_CONFIGS, PREFIX)
