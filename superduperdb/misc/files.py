import hashlib


def _get_file(uri):
    """
    Get file name from uri.

    >>> _get_file('file://test.txt')
    'test.txt'
    >>> _get_file('http://test.txt')
    '414388bd5644669b8a92e45a96318890f6e8de54'
    """
    if uri.startswith('file://'):
        file = uri[7:]
    elif (
        uri.startswith('http://')
        or uri.startswith('https://')
        or uri.startswith('s3://')
    ):
        file = hashlib.sha1(uri.encode()).hexdigest()
    else:
        raise NotImplementedError(f'File type of {file} not supported')
    return file


def load_uris(r: dict, root: str, raises: bool = False):
    """
    Load ``"bytes"`` into ``"_content"`` from ``"uri"`` inside ``r``.

    :param r: The dict to load the bytes into
    :param root: The root directory to load the bytes from
    :param raises: Whether to raise an error if the file is not found

    >>> with open('/tmp/test.txt', 'wb') as f:
    ...     _ = f.write(bytes('test', 'utf-8'))
    >>> r = {"_content": {"uri": "file://test.txt"}}
    >>> load_uris(r, '/tmp')
    >>> r
    {'_content': {'uri': 'file://test.txt', 'bytes': b'test'}}
    """
    for k, v in r.items():
        if isinstance(v, dict):
            if k == '_content' and 'uri' in v and 'bytes' not in v:
                file = _get_file(v['uri'])
                if root:
                    file = f'{root}/{file}'
                try:
                    with open(f'{file}', 'rb') as f:
                        r['_content']['bytes'] = f.read()
                except FileNotFoundError as e:
                    if raises:
                        raise e
            else:
                load_uris(v, root)
        elif isinstance(v, list):
            for i in v:
                load_uris(i, root)
        else:
            pass
