**`superduperdb.backends.base.metadata`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/backends/base/metadata.py)

## `MetaDataStore` 

```python
MetaDataStore(self,
     conn: Any,
     name: Optional[str] = None)
```
| Parameter | Description |
|-----------|-------------|
| conn | connection to the meta-data store |
| name | Name to identify DB using the connection |

Abstraction for storing meta-data separately from primary data.

## `NonExistentMetadataError` 

```python
NonExistentMetadataError(self,
     /,
     *args,
     **kwargs)
```
| Parameter | Description |
|-----------|-------------|
| args | *args for `Exception` |
| kwargs | **kwargs for `Exception` |

NonExistentMetadataError.

