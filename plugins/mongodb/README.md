<!-- Auto-generated content start -->
# superduper_mongodb

SuperDuper MongoDB is a Python library that provides a high-level API for working with MongoDB. It is built on top of pymongo and provides a more user-friendly interface for working with MongoDB.

In general the MongoDB query API works exactly as per pymongo, with the exception that:

- inputs are wrapped in Document
- additional support for vector-search is provided
- queries are executed lazily


## Installation

```bash
pip install superduper_mongodb
```

## API


- [Code](https://github.com/superduper-io/superduper/tree/main/plugins/mongodb)
- [API-docs](/docs/api/plugins/superduper_mongodb)

| Class | Description |
|---|---|
| `superduper_mongodb.data_backend.MongoDataBackend` | Data backend for MongoDB. |
| `superduper_mongodb.metadata.MongoMetaDataStore` | Metadata store for MongoDB. |
| `superduper_mongodb.query.MongoQuery` | A query class for MongoDB. |
| `superduper_mongodb.query.BulkOp` | A bulk operation for MongoDB. |
| `superduper_mongodb.artifacts.MongoArtifactStore` | Artifact store for MongoDB. |



<!-- Auto-generated content end -->

<!-- Add your additional content below -->

## Connection examples

### Connect to mongomock
```python
from superduper import superduper
db = superduper('mongomock://test')
```

### Connect to MongoDB
```python
from superduper import superduper
db = superduper('mongodb://localhost:27017/documents')
```

### Connect to MongoDB Atlas
```python
from superduper import superduper
db = superduper('mongodb+srv://<username>:<password>@<cluster-url>/<database>')
```

## Query examples

### Inserts

```python
db['my-collection'].insert_many([{'my-field': ..., ...}
    for _ in range(20)
]).execute()
```

### Updates

```python
db['my-collection'].update_many(
    {'<my>': '<filter>'},
    {'$set': ...},
).execute()
```

### Selects

```python
db['my-collection'].find({}, {'_id': 1}).limit(10).execute()
```

### Vector-search

Vector-searches may be integrated with `.find`.

```python
db['my-collection'].like({'img': <my_image>}, vector_index='my-index-name').find({}, {'img': 1}).execute()
```

Read more about vector-search [here](../fundamentals/vector_search_algorithm.md).

### Deletes

```python
db['my-collection'].delete_many({}).execute()
```

