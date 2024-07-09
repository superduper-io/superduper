**`superduper.backends.base.artifacts`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper/backends/base/artifacts.py)

## `ArtifactSavingError` 

```python
ArtifactSavingError(self,
     /,
     *args,
     **kwargs)
```
| Parameter | Description |
|-----------|-------------|
| args | *args for `Exception` |
| kwargs | **kwargs for `Exception` |

Error when saving artifact in artifact store fails.

## `ArtifactStore` 

```python
ArtifactStore(self,
     conn: Any,
     name: Optional[str] = None)
```
| Parameter | Description |
|-----------|-------------|
| conn | connection to the meta-data store |
| name | Name to identify DB using the connection |

Abstraction for storing large artifacts separately from primary data.

