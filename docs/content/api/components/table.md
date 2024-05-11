**`superduperdb.components.table`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/components/table.py)

## `Table` 

```python
Table(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     *,
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     schema: superduperdb.components.schema.Schema,
     primary_id: str = 'id') -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| artifacts | A dictionary of artifacts paths and `DataType` objects |
| schema | The schema of the table |
| primary_id | The primary id of the table |

A component that represents a table in a database.

