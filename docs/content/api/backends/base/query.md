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
     documents,
     builder_cls,
     db: Optional[ForwardRef('Datalayer')] = None)
```
| Parameter | Description |
|-----------|-------------|
| query | The query to parse. |
| documents | The documents to query. |
| builder_cls | The class to use to build the query. |
| db | The datalayer to use to execute the query. |

Parse a string query into a query object.

## `Model` 

```python
Model(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |

A model helper class for create a query to predict.

## `PredictOne` 

```python
PredictOne(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     args: Sequence = <factory>,
     kwargs: Dict = <factory>) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| args | The arguments to pass to the model |
| kwargs | The keyword arguments to pass to the model |

A query to predict a single document.

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

