---
sidebar_position: 4
---
# Chunked

## Chunked vector-search using multiple inputs per document

In this example, we demonstrate how to implement vector-search, where the targets of search are envisaged
to be smaller units of text than the raw data. For example, a developer might like to store whole documents, 
but search within those documents, finding substrings and references to the original document. 

This workflow is much trickier to implement than vanilla vector-search. Luckily, with `superduperdb`, it is just one extra command.


```python
import os

# Uncomment one of the following lines to use a bespoke MongoDB deployment
# For testing the default connection is to mongomock

mongodb_uri = os.getenv("MONGODB_URI", "mongomock://test")
# mongodb_uri = "mongodb://localhost:27017"
# mongodb_uri = "mongodb://superduper:superduper@mongodb:27017/documents"
# mongodb_uri = "mongodb://<user>:<pass>@<mongo_cluster>/<database>"
# mongodb_uri = "mongodb+srv://<username>:<password>@<atlas_cluster>/<database>"

# Super-Duper your Database!
from superduperdb import superduper
db = superduper(mongodb_uri)
```

To demonstrate this type of search with larger chunks of text, we use a wikipedia sample.


```python
!curl -O https://superduperdb-public.s3.eu-west-1.amazonaws.com/wikipedia-sample.json
```

As before we insert the data using `pymongo`-similar syntax:


```python
import json
from superduperdb.backends.mongodb import Collection
from superduperdb import Document as D

with open('wikipedia-sample.json') as f:
    data = json.load(f)[:100]

db.execute(Collection('wikipedia').insert_many([D(r) for r in data]))
```

Let's have a look at a document:


```python
r = db.execute(Collection('wikipedia').find_one()).unpack()
r
```

To create the search functionality, we set up a simple model, whose sole purpose is to chunk 
the raw text into parts, and save those parts in another collecion:


```python
from superduperdb import Model

def splitter(r):
    out = [r['title']]
    split = r['abstract'].split(' ')
    for i in range(0, len(split) - 5, 5):
        out.append(' '.join(split[i: i + 5]))
    out = [x for x in out if x]
    return out


model = Model(
    identifier='splitter',
    object=splitter,
    flatten=True,
    model_update_kwargs={'document_embedded': False},
)

model.predict(r, one=True)
```

Let's apply this model to the whole input collection:


```python
model.predict(
    X='_base', 
    db=db,
    select=Collection('wikipedia').find()
)
```

Now let's look at the split data:


```python
db.execute(Collection('_outputs._base.splitter').find_one())
```

We can search this data in a manner similar to previously:


```python
from superduperdb import VectorIndex, Listener
from superduperdb.ext.openai import OpenAIEmbedding

model = OpenAIEmbedding(model='text-embedding-ada-002')

db.add(
    VectorIndex(
        identifier=f'chunked-documents',
        indexing_listener=Listener(
            model=model,
            key='_outputs._base.splitter',
            select=Collection('_outputs._base.splitter').find(),
            predict_kwargs={'max_chunk_size': 1000},
        ),
        compatible_listener=Listener(
            model=model,
            key='_base',
            select=None,
            active=False,
        )
    )
)
```

Now we can search through the split-text collection and recall the full original documents,
highlighting which text was found to be relevant:


```python
from superduperdb.backends.mongodb import Collection
from superduperdb import Document as D
from IPython.display import *

query = 'politics'

shingle_collection = Collection('_outputs._base.splitter')
main_collection = Collection('wikipedia')

result = db.execute(
    shingle_collection
        .like(D({'_base': query}), vector_index='chunked-documents', n=5)
        .find({}, {'_outputs._base.text-embedding-ada-002': 0})
)

display(Markdown(f'---'))
for shingle in result:
    original = db.execute(main_collection.find_one({'_id': shingle['_source']}))

    display(Markdown(f'# {original["title"]}"'))
    
    start = original['abstract'].find(shingle['_outputs']['_base']['splitter'])

    to_format = (
        original["abstract"][:start] + '**' + '<span style="color:red">' +
        shingle["_outputs"]["_base"]["splitter"].upper() + '**' + '<span style="color:black">' +
        original["abstract"][start + len(shingle["_outputs"]["_base"]["splitter"]):]
    )
    
    display(Markdown(to_format))
    display(Markdown(f'---'))
```
