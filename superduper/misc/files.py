import hashlib
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
            logging.info(f'Successfully loaded secret {key_dir} into environment.')
        except Exception as e:
            logging.error(f"Error reading file {secret_file_path}: {e}")


def get_file_from_uri(uri):
    """
    Get file name from uri.

    >>> _get_file('file://test.txt')
    'test.txt'
    >>> _get_file('http://test.txt')
    '414388bd5644669b8a92e45a96318890f6e8de54'

    :param uri: The uri to get the file from
    """
    if uri.startswith('file://'):
        file = uri[7:]
    elif (
        uri.startswith('http://')
        or uri.startswith('https://')
        or uri.startswith('s3://')
    ):
        file = f'{CFG.downloads.folder}/{hashlib.sha1(uri.encode()).hexdigest()}'
    else:
        raise NotImplementedError(f'File type of {uri} not supported')
    return file
