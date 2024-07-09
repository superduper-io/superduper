**`superduper.backends.sqlalchemy.metadata`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper/backends/sqlalchemy/metadata.py)

## `SQLAlchemyMetadata` 

```python
SQLAlchemyMetadata(self,
     conn: Any,
     name: Optional[str] = None)
```
| Parameter | Description |
|-----------|-------------|
| conn | connection to the meta-data store |
| name | Name to identify DB using the connection |

Abstraction for storing meta-data separately from primary data.

