**`superduperdb.backends.ibis.field_types`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/backends/ibis/field_types.py)

## `dtype` 

```python
dtype(x)
```
| Parameter | Description |
|-----------|-------------|
| x | The data type e.g int, str, etc. |

Ibis dtype to represent basic data types in ibis.

## `FieldType` 

```python
FieldType(self,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     *,
     identifier: Union[str,
     ibis.expr.datatypes.core.DataType]) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | The name of the data type. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |

Field type to represent the type of a field in a table.

This is a wrapper around ibis.expr.datatypes.DataType to make it
serializable.

