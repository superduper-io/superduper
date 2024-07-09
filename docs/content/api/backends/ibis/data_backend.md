**`superduper.backends.ibis.data_backend`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper/backends/ibis/data_backend.py)

## `IbisDataBackend` 

```python
IbisDataBackend(self,
     conn: ibis.backends.base.BaseBackend,
     name: str,
     in_memory: bool = False)
```
| Parameter | Description |
|-----------|-------------|
| conn | The connection to the database. |
| name | The name of the database. |
| in_memory | Whether to store the data in memory. |

Ibis data backend for the database.

