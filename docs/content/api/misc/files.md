**`superduperdb.misc.files`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/misc/files.py)

## `load_uris` 

```python
load_uris(r: dict,
     datatypes: Dict,
     root: Optional[str] = None,
     raises: bool = False)
```
| Parameter | Description |
|-----------|-------------|
| r | The dict to load the bytes into |
| datatypes | The datatypes to use for encoding |
| root | The root directory to load the bytes from |
| raises | Whether to raise an error if the file is not found |

Load ``"bytes"`` into ``"_content"`` from ``"uri"`` inside ``r``.

```python
with open('/tmp/test.txt', 'wb') as f:
    _ = f.write(bytes('test', 'utf-8'))
r = {"_content": {"uri": "file://test.txt"}}
load_uris(r, '/tmp')
r
# {'_content': {'uri': 'file://test.txt', 'bytes': b'test'}}
```

## `get_file_from_uri` 

```python
get_file_from_uri(uri)
```
| Parameter | Description |
|-----------|-------------|
| uri | The uri to get the file from |

Get file name from uri.

```python
_get_file('file://test.txt')
# 'test.txt'
_get_file('http://test.txt')
# '414388bd5644669b8a92e45a96318890f6e8de54'
```

