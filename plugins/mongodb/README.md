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