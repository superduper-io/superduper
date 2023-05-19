from . import config
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from superduperdb.misc import dicts
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import os

Self = Any
File = Union[Path, str]

ROOT = Path(__file__).parents[1]
PREFIX = 'SUPERDUPERDB_'
FILES_NAME = 'CONFIG_FILES'
DEFAULT_FILES = str(ROOT / 'configs.json')
FILE_SEP = ','


@dataclass(frozen=True)
class ConfigSettings:
    cls: Type
    default_files: str
    prefix: str
    environ: Optional[Dict] = None

    @cached_property
    def config(self) -> Any:
        """Read a Pydantic class"""
        environ = dict(os.environ if self.environ is None else self.environ)

        file_names = environ.pop(self.prefix + FILES_NAME, self.default_files)
        data = dicts.read_all(file_names.split(FILE_SEP))
        parent = self.cls().dict()
        environ_dict = dicts.environ_to_config_dict(self.prefix, parent, environ)
        return self.cls(**dicts.combine((*data, environ_dict)))


CONFIG = ConfigSettings(config.Config, DEFAULT_FILES, PREFIX)
