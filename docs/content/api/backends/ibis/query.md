**`superduperdb.backends.ibis.query`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/backends/ibis/query.py)

## `parse_query` 

```python
parse_query(query,
     documents: Sequence[Dict] = (),
     db: Optional[ForwardRef('Datalayer')] = None)
```
| Parameter | Description |
|-----------|-------------|
| query | The query to parse. |
| documents | The documents to query. |
| db | The datalayer to use to execute the query. |

Parse a string query into a query object.

## `IbisQuery` 

```python
IbisQuery(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     *,
     parts: Sequence[Union[Tuple,
     str]] = ()) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| parts | The parts of the query. |

A query that can be executed on an Ibis database.

## `RawSQL` 

```python
RawSQL(self,
     query: str,
     id_field: str = 'id') -> None
```
| Parameter | Description |
|-----------|-------------|
| query | The raw SQL query |
| id_field | The field to use as the primary id |

Raw SQL query.

