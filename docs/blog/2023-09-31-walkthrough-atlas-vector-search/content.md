# A walkthrough of vector-search on MongoDB Atlas with SuperDuperDB

*In this tutorial we show developers how to execute searches leveraging MongoDB Atlas vector-search
via SuperDuperDB*

## Step 1: install `superduperdb` Python package

```
pip install superduperdb
```

## Step 2: connect to your Atlas cluster using SuperDuperDB

```python
import pymongo
from superduperdb import superduper

URI = 'mongodb://<your-connection-string-here>'

db = pymongo.MongoClient(URI).my_database

db = superduper(db)
```

## Step 1: insert some data into your Atlas cluster

```python
from superduperdb.db.mongodb.query import Collection

collection = Collection('documents')

db.execute(
    collection.insert_many([
        Document(r) for r in data
    ])
)
```