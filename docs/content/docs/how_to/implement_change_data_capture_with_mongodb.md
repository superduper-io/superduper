# Perform change data capture (CDC)

In a standalone MongoDB deployment, users are required to insert data directly through the 
SuperDuperDB `Datalayer` or client (which triggers the `Datalayer`). For use-cases 
with multiple users, stakeholders, and potentially automated data-updates on the database,
this is not sufficient. For that reason SuperDuperDB supports a paradigm known as 
change-data-capture (CDC). 

In change-data-capture, a service is deployed which listeners the data deployment for changes, and 
reacts to these changes, activating models which are configured to compute outputs over new data.

In this notebook, we demonstrate how to use CDC with SuperDuperDB.

The notebook requires that a MongoDB replica set has been set up.

```python
import pymongo
import sys

sys.path.append('../')

from superduperdb.ext.numpy.array import array
from superduperdb.db.mongodb.query import Collection
from superduperdb import superduper
from superduperdb.container.listener import Listener
from superduperdb.container.vector_index import VectorIndex
from superduperdb.db.base.cdc import DatabaseListener
from superduperdb.container.document import Document as D
```


```python
db = pymongo.MongoClient().documents
db = superduper(db)

collection = Collection('cdc_example')
```

Insert the data into `documents` collection


```python
data = [
  {
    "title": "Politics of Armenia",
    "abstract": "The politics of Armenia take place in the framework of the parliamentary representative democratic republic of Armenia, whereby the President of Armenia is the head of state and the Prime Minister of Armenia the head of government, and of a multi-party system. Executive power is exercised by the President and the Government."
  },
  {
    "title": "Foreign relations of Armenia",
    "abstract": "Since its independence, Armenia has maintained a policy of complementarism by trying to have positive and friendly relations with Iran, Russia, and the West, including the United States and the European Union.– \"Armenian Foreign Policy Between Russia, Iran And U."
  },
  {
    "title": "Demographics of American Samoa",
    "abstract": "This article is about the demographics of American Samoa, including population density, ethnicity, education level, health of the populace, economic status, religious affiliations and other aspects of the population. American Samoa is an unincorporated territory of the United States located in the South Pacific Ocean."
  },
  {
    "title": "Analysis",
    "abstract": "Analysis is the process of breaking a complex topic or substance into smaller parts in order to gain a better understanding of it. The technique has been applied in the study of mathematics and logic since before Aristotle (384–322 B."
  }
]

data = [D(d) for d in data]

db.execute(collection.insert_many(data))
```

Create a vector index listener.
This consist a indexing listener (SentenceTransformer) model to vectorize a key.


```python
import sentence_transformers 
from superduperdb.container.model import Model

model = Model(
    identifier='all-MiniLM-L6-v2',
    object=sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2'),
    encoder=array('float32', shape=(384,)),
    predict_method='encode',
    batch_predict=True,
)

db.add(VectorIndex(
    identifier='my-index',
    indexing_listener=Listener(
        model=model,
        key='abstract',
        select=Collection(name='documents').find()
    ),
))
```

Create instance of `DatabaseListener` and start listening the `documents` collection.


```python
database_listener = DatabaseListener(
    db=db,
    identifier='basic-cdc-listener',
    on=collection,
)
database_listener.listen()
```

Check the listener's status


```python
database_listener.is_available()
```

You can check information stored by the listener.


```python
database_listener.info()
```

Add 2 documents and check the info again


```python
data = [
    {
        "title": "Politics of India",
        "abstract": "Some despriction 1",
    }, 
    {
        "title": "Politics of Asia",
        "abstract": "some description 2",
    }
]
doc = db_mongo.test_db.documents.insert_many(data)
```

Check the inserts info again


```python
database_listener.info()
```

Check that the vectors synced between LanceDB and MongoDB


```python
from superduperdb.vector_search.lancedb_client import LanceDBClient
from superduperdb import CFG
```


```python
client = db.vector_database.client
```

Use the identifier to extract the correct table in LanceDB (`<model>/<key>`)


```python
table = client.get_table('test-st/abstract')
```


```python
table.table.to_pandas()
```
