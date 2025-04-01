import os
import typing as t
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import warnings
from warnings import warn

import yaml

from . import config_dicts
from .config import _dataclass_from_dict

# Type definitions
File = t.Union[Path, str]

# Environment and path constants
HOME = os.environ.get('HOME', '')
PREFIX = 'SUPERDUPER_'
CONFIG_FILE = os.environ.get(f'{PREFIX}CONFIG')
USER_CONFIG: t.Optional[str] = (
    str(Path(CONFIG_FILE).expanduser())
    if CONFIG_FILE
    else (f'{HOME}/.superduper/config.yaml' if HOME else None)
)
ROOT = Path(__file__).parents[2]


class ConfigError(Exception):
    """
    An exception raised when there is an error in the configuration.

    Args:
        message: Error message explaining the configuration issue
        source: Optional source of the error (e.g., "file", "environment")
    """

    def __init__(self, message: str, source: t.Optional[str] = None):
        self.source = source
        super().__init__(message)


def load_secrets(secrets_dir: t.Optional[str] = None) -> None:
    """Load secrets from a directory into environment variables.

    Each subdirectory name becomes an environment variable name (converted to
    uppercase with dashes replaced by underscores), and the contents of the
    'secret_string' file in that subdirectory becomes the value.

    Args:
        secrets_dir: The directory containing the secrets subdirectories.
                    If None, uses the value from the global configuration.

    Returns:
        None

    Warns:
        UserWarning: If the secrets directory doesn't exist or if any subdirectory
                     is missing a 'secret_string' file.
    """
    if secrets_dir is None:
        from superduper import CFG
        secrets_dir = CFG.secrets_volume

    secrets_dir = os.path.expanduser(secrets_dir)

    if not os.path.isdir(secrets_dir):
        warn(f"Warning: The secrets path '{secrets_dir}' is not a valid directory.")
        return

    try:
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

            # Store the secret in an environment variable.
            # To do that, we need to replace secrets naming pattern with envars pattern.
            # Example: /session/secrets/aws-secret-access-key to AWS_SECRET_ACCESS_KEY
            env_name = key_dir.replace('-', '_').upper()
            os.environ[env_name] = content
    except (IOError, OSError) as e:
        warn(f"Warning: Failed to read secrets directory {secrets_dir}: {str(e)}")


@dataclass(frozen=True)
class ConfigSettings:
    """Helper class to read configuration from files and environment variables.

    This class handles loading configuration values from multiple sources with the
    following precedence (highest to lowest):
    1. Environment variables
    2. User configuration file
    3. Default values from the dataclass

    Args:
        cls: The dataclass type to instantiate with the configuration values
        environ: Custom environment variables dictionary (uses os.environ if None)
        base: The base field name in the configuration file (e.g., "cluster" loads from r["cluster"])
    """

    cls: t.Type
    environ: t.Optional[t.Dict[str, str]] = None
    base: t.Optional[str] = None

    @cached_property
    def config(self) -> t.Any:
        """Read configuration using defined precedence rules.

        Returns:
            An instance of the specified dataclass populated with configuration values

        Raises:
            ConfigError: If the specified config file doesn't exist (unless it's the default)
        """
        # Start with defaults from the class
        parent = self.cls().dict()

        # Process environment variables
        env = dict(os.environ if self.environ is None else self.environ)
        prefix = PREFIX
        if self.base:
            prefix = f"{PREFIX}{self.base.upper()}_"

        env = config_dicts.environ_to_config_dict(prefix, parent, env)

        # Handle secrets if configured
        secrets_volume = env.get('secrets_volume') or parent.get('secrets_volume')
        if secrets_volume:
            secrets_volume = os.path.expanduser(secrets_volume)
            if os.path.isdir(secrets_volume):
                load_secrets(secrets_volume)
                # Refresh environment variables after loading secrets
                env = config_dicts.environ_to_config_dict(
                    prefix, parent, dict(os.environ if self.environ is None else self.environ)
                )
            else:
                warn(f"Warning: The secrets path '{secrets_volume}' is not a valid directory.")

        # Load configuration from file
        file_config: t.Dict[str, t.Any] = {}
        if USER_CONFIG is not None:
            try:
                with open(USER_CONFIG) as f:
                    file_config = yaml.safe_load(f) or {}
            except FileNotFoundError as e:
                if USER_CONFIG != f'{HOME}/.superduper/config.yaml':
                    raise ConfigError(
                        f'Could not find config file: {USER_CONFIG}',
                        source='file'
                    ) from e
                # Default config file is allowed to be missing
            except yaml.YAMLError as e:
                raise ConfigError(
                    f'Invalid YAML in config file: {USER_CONFIG}',
                    source='file'
                ) from e

            # Extract section if base is specified
            if self.base and file_config:
                file_config = file_config.get(self.base, {})

        # Combine all configuration sources with proper precedence
        kwargs = config_dicts.combine_configs((parent, file_config, env))

        # Convert to the target dataclass
        return _dataclass_from_dict(self.cls, kwargs)