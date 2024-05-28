**`superduperdb.misc.serialization`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/misc/serialization.py)

## `asdict` 

```python
asdict(obj,
     *,
     copy_method=<function copy at 0x100451ee0>) -> Dict[str,
     Any]
```
| Parameter | Description |
|-----------|-------------|
| obj | The dataclass instance to |
| copy_method | The copy method to use for non atomic objects |

Convert the dataclass instance to a dict.

Custom ``asdict`` function which exports a dataclass object into a dict,
with a option to choose for nested non atomic objects copy strategy.

