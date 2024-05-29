**`superduperdb.backends.base.query`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/backends/base/query.py)

## `applies_to` 

```python
applies_to(*flavours)
```
| Parameter | Description |
|-----------|-------------|
| flavours | The flavours to check against. |

Decorator to check if the query matches the accepted flavours.

## `parse_query` 

```python
parse_query(query: Union[str,
     list],
     builder_cls: Optional[Type[superduperdb.backends.base.query.Query]] = None,
     documents: Sequence[Any] = (),
     db: Optional[ForwardRef('Datalayer')] = None)
```
| Parameter | Description |
|-----------|-------------|
| query | The query to parse. |
| builder_cls | The class to use to build the query. |
| documents | The documents to query. |
| db | The datalayer to use to execute the query. |

Parse a string query into a query object.

## `Model` 

```python
Model(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     parts: Sequence[Union[Tuple,
     str]] = ()) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| parts | The parts of the query. |

A model helper class for create a query to predict.

## `Query` 

```python
Query(self,
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

A query object.

This base class is used to create a query object that can be executed
in the datalayer.

## `TraceMixin` 

```python
TraceMixin(self,
     /,
     *args,
     **kwargs)
```
Mixin to add trace functionality to a query.

