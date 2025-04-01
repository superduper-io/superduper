import os
import typing as t
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from warnings import warn

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


def load_secrets(secrets_dir: str | None = None):
    """Load secrets directory into env vars.

    :param secrets_dir: The directory containing the secrets.
    """
    if secrets_dir is None:
        from superduper import CFG

        secrets_dir = CFG.secrets_volume

    if not os.path.isdir(secrets_dir):
        warn(f"Warning: The path '{secrets_dir}' is not a valid directory.")

    for key_dir in os.listdir(secrets_dir):
        key_path = os.path.join(secrets_dir, key_dir)

        if not os.path.isdir(key_path):
            continue

        secret_file_path = os.path.join(key_path, 'secret_string')

        if not os.path.isfile(secret_file_path):
            warn(f"Warning: No 'secret_string' file found in {key_path}.")
            continue

        with open(secret_file_path, 'r') as file:
            content = file.read().strip()
        env_name = key_dir.replace('-', '_').upper()
        os.environ[env_name] = content


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

        secrets_volume = env.get('secrets_volume') or parent.get('secrets_volume')

        if secrets_volume:
            secrets_volume = os.path.expanduser(secrets_volume)

        if secrets_volume and os.path.isdir(secrets_volume):
            load_secrets(secrets_volume)
            env = config_dicts.environ_to_config_dict(
                prefix, parent, dict(os.environ if os.environ else {})
            )
        elif secrets_volume:
            warn(f"Warning: The path '{secrets_volume}' is not a valid directory.")

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
