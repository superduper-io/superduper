---
sidebar_position: 2
---

# MongoDB 

In general the MongoDB query API works exactly as per `pymongo`, with the exception that:

- inputs are wrapped in `Document`
- additional support for vector-search is provided
- queries are executed lazily

## Inserts

```python
db['my-collection'].insert_many([{'my-field': ..., ...}
    for _ in range(20)
]).execute()
```

## Updates

```python
db['my-collection'].update_many(
    {'<my>': '<filter>'},
    {'$set': ...},
).execute()
```

## Selects

```python
db['my-collection'].find({}, {'_id': 1}).limit(10).execute()
```

## Vector-search

Vector-searches may be integrated with `.find`.

```python
db['my-collection'].like({'img': <my_image>}, vector_index='my-index-name').find({}, {'img': 1}).execute()
```

Read more about vector-search [here](../fundamentals/vector_search_algorithm.md).

## Deletes

```python
db['my-collection'].delete_many({}).execute()
```