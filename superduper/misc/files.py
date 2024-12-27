import hashlib
import os

from superduper import CFG


def load_secrets():
    """Help method to load secrets from directory."""
    secrets_dir = CFG.secrets_volume
    if not os.path.isdir(secrets_dir):
        raise ValueError(f"The path '{secrets_dir}' is not a valid directory.")

    for root, _, files in os.walk(secrets_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                with open(file_path, 'r') as file:
                    content = file.read().strip()

                key = file_name
                os.environ[key] = content
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")


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
