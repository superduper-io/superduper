**`superduperdb.rest.utils`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/rest/utils.py)

## `parse_query` 

```python
parse_query(query,
     documents,
     db)
```
| Parameter | Description |
|-----------|-------------|
| query | query string to parse |
| documents | documents to use in the query |
| db | datalayer instance |

Parse a query string into a query object.

## `strip_artifacts` 

```python
strip_artifacts(r: Any)
```
| Parameter | Description |
|-----------|-------------|
| r | the data to strip artifacts from |

Strip artifacts for the data.

