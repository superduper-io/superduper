# A walkthrough of vector-search on MongoDB Atlas with SuperDuperDB

*In this tutorial we show developers how to execute searches leveraging MongoDB Atlas vector-search
via SuperDuperDB*

**Install `superduperdb` Python package**

Using vector-search with SuperDuperDB on MongoDB requires only one simple python package install:

```bash
pip install superduperdb
```

With this install SuperDuperDB includes all the packages needed to define a range of API based and package based 
vector-search models, such as OpenAI and Hugging-Face's `transformers`.

**Connect to your Atlas cluster using SuperDuperDB**

SuperDuperDB ships with it's own MongoDB python client, which supports
all commands supported by `pymongo`. In the example below 
the key to connecting to your Atlas cluster is the `db` object.

The `db` object contains all functionality needed to read and write to 
the MongoDB instance and also to define, save and apply a flexible range 
of AI models for vector-search.

```python
from superduperdb import CFG
from superduperdb.db.base.build import build_datalayer

URI = 'mongodb://<your-atlas-connection-string-here>'

CFG.vector_search = URI
CFG.databackend = URI

db = build_datalayer()
```

**Load your data**

You can download some data to play with from [this link](https://superduperdb-public.s3.eu-west-1.amazonaws.com/pymongo.json):

```bash
curl -O https://superduperdb-public.s3.eu-west-1.amazonaws.com/pymongo.json
```

The data contains all inline doc-strings of the `pymongo` Python API (official
MongoDB driver for Python). The name of the function or class is in `"res"` and
the doc-string is contained in `"value"`.

```python
import json

with open('pymongo.json') as f:
    data = json.load(f)
```

Here's one record to illustrate the data:

```json
{
  "key": "pymongo.mongo_client.MongoClient",
  "parent": null,
  "value": "\nClient for a MongoDB instance, a replica set, or a set of mongoses.\n\n",
  "document": "mongo_client.md",
  "res": "pymongo.mongo_client.MongoClient"
}
```

**Insert the data into your Atlas cluster**

We can use the SuperDuperDB connection to insert this data

```python
from superduperdb.db.mongodb.query import Collection

collection = Collection('documents')

db.execute(
    collection.insert_many([
        Document(r) for r in data
    ])
)
```

**Define your vector model and vector-index**

Now we have data in our collection we can define the vector-index:

```python
from superduperdb.container.vector_index import VectorIndex
from superduperdb.container.listener import Listener
from superduperdb.ext.numpy.array import array
from superduperdb.ext.openai.model import OpenAIEmbedding

model = OpenAIEmbedding(model='text-embedding-ada-002')

db.add(
    VectorIndex(
        identifier=f'pymongo-docs',
        indexing_listener=Listener(
            model=model,
            key='value',
            select=Collection('documents').find(),
            predict_kwargs={'max_chunk_size': 1000},
        ),
    )
)
```

This command tells the system that we want to:

- search the `"documents"` collection
- set-up a vector-index on our Atlas cluster, using the text in the `"value"` field
- use the OpenAI model `"text-embedding-ada-002"` to create vector-embeddings

After issuing this command, SuperDuperDB does these things:

- Configures an MongoDB Atlas knn-index in the `"documents"` collection
- Saves the `model` object in the SuperDuperDB model store hosted on `gridfs`
- Applies `model` to all data in the `"documents"` collection, and saves the vectors in the documents
- Saves the fact that `model` is connected to the `"pymongo-docs"` vector-index

**Use vector-search in a super-duper query**

Now we are ready to use the SuperDuperDB query-API for vector-search.
You'll see below, that SuperDuperDB handles all logic related to 
converting queries on the fly to vectors under the hood.

```python
from superduperdb.db.mongodb.query import Collection
from superduperdb.container.document import Document as D
from IPython.display import *

query = 'Find data'

result = db.execute(
    Collection('documents')
        .like(D({'value': query}), vector_index='pymongo-docs', n=5)
        .find()
)

for r in result:
    display(Markdown(f'### `{r["parent"] + "." if r["parent"] else ""}{r["res"]}`'))
    display(Markdown(r['value']))
```