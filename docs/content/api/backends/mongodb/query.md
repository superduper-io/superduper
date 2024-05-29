**`superduperdb.backends.mongodb.query`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/backends/mongodb/query.py)

## `DeleteOne` 

```python
DeleteOne(**kwargs)
```
| Parameter | Description |
|-----------|-------------|
| kwargs | The arguments to pass to the operation. |

DeleteOne operation for MongoDB.

## `InsertOne` 

```python
InsertOne(**kwargs)
```
| Parameter | Description |
|-----------|-------------|
| kwargs | The arguments to pass to the operation. |

InsertOne operation for MongoDB.

## `ReplaceOne` 

```python
ReplaceOne(**kwargs)
```
| Parameter | Description |
|-----------|-------------|
| kwargs | The arguments to pass to the operation. |

ReplaceOne operation for MongoDB.

## `UpdateOne` 

```python
UpdateOne(**kwargs)
```
| Parameter | Description |
|-----------|-------------|
| kwargs | The arguments to pass to the operation. |

UpdateOne operation for MongoDB.

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

## `MongoQuery` 

```python
MongoQuery(self,
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

A query class for MongoDB.

This class is used to build and execute queries on a MongoDB database.

## `BulkOp` 

```python
BulkOp(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     *,
     kwargs: Dict = <factory>) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| kwargs | The arguments to pass to the operation. |

A bulk operation for MongoDB.

## `ChangeStream` 

```python
ChangeStream(self,
     collection: str,
     args: Sequence = <factory>,
     kwargs: Dict = <factory>) -> None
```
| Parameter | Description |
|-----------|-------------|
| collection | The collection to perform the query on |
| args | Positional query arguments to ``pymongo.Collection.watch`` |
| kwargs | Named query arguments to ``pymongo.Collection.watch`` |

Change stream class to watch for changes in specified collection.

