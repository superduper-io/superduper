import hashlib


def load_uris(r: dict, root: str, raises: bool = False):
    """
    Load ``"bytes"`` into ``"_content"`` from ``"uri"`` inside ``r``.

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
                if v['uri'].startswith('file://'):
                    file = v['uri'][7:]
                elif (
                    v['uri'].startswith('http://')
                    or v['uri'].startswith('https://')
                    or v['uri'].startswith('s3://')
                ):
                    file = hashlib.sha1(v['uri'].encode()).hexdigest()
                else:
                    raise NotImplementedError(f'File type of {v} not supported')

                try:
                    if root:
                        file = f'{root}/{file}'
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
