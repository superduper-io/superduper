---
sidebar_position: 2
---

# MongoDB 

In general the MongoDB query API works exactly as `pymongo`, with the exception that:

- inputs are wrapped in `Document`
- additional support for vector-search is provided

## Inserts

```python
from superduperdb.backends.mongodb import Collection

collection = Collection('my-collection-name')

db.execute(
    collection.insert_many([
        Document({'my-field': ..., ...})
        for _ in range(20)
    ])
)
```

## Updates

```python
db.execute(
    collection.update_many(
        {'<my>': '<filter>'},
        Document({'$set': ...})
    )
)
```

## Selects

```python
db.execute(
    collection.find({}, {'_id': 1}).limit(10)
)
```

### Vector-search

Vector-searches are a subset of standard select statements.
They may be integrated with `.find`.

```python
from superduperdb import Document
db.execute(
    collection.like(Document({'img': <my_image>}), vector_index='my-index-name').find({}, {'img': 1})
)
```

Read more about vector-search [here](../fundamentals/vector_search_algorithm.md).

## Deletes

```python
db.execute(collection.delete_many({}))
```

## Aggregate

Aggregates are exactly as in `pymongo`, with the exception that a `$vectorSearch` stage may be
fed with an additional field `'like': Document({...})`, which plays the same role as in selects.

Read more about this in [the vector-search section](../walkthrough/vector_search).
