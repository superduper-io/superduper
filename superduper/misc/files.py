# TODO move to services
import os

from superduper import CFG, logging


def load_secrets():
    """Load secrets directory into env vars."""
    secrets_dir = CFG.secrets_volume

    if not os.path.isdir(secrets_dir):
        raise ValueError(f"The path '{secrets_dir}' is not a valid secrets directory.")

    for key_dir in os.listdir(secrets_dir):
        key_path = os.path.join(secrets_dir, key_dir)

        if not os.path.isdir(key_path):
            continue

        secret_file_path = os.path.join(key_path, 'secret_string')

        if not os.path.isfile(secret_file_path):
            logging.warn(f"Warning: No 'secret_string' file found in {key_path}.")
            continue

        try:
            with open(secret_file_path, 'r') as file:
                content = file.read().strip()

            os.environ[key_dir] = content
        except Exception as e:
            logging.error(f"Error reading file {secret_file_path}: {e}")
