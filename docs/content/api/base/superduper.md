**`superduperdb.base.superduper`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/base/superduper.py)

## `superduper` 

```python
superduper(item: Optional[Any] = None,
     **kwargs) -> Any
```
| Parameter | Description |
|-----------|-------------|
| item | A database or model |
| kwargs | Additional keyword arguments to pass to the component |

Superduper API to automatically wrap an object to a db or a component.

Attempts to automatically wrap an item in a superduperdb component by
using duck typing to recognize it.

