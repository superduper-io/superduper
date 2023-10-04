# MongoDB Atlas vector-search with SuperDuperDB


```python
!pip install superduperdb
```


```python
!curl -O https://superduperdb-public.s3.eu-west-1.amazonaws.com/pymongo.json
```


```python
import os

os.environ['OPENAI_API_KEY'] = '<YOUR-OPEN-AI-API-KEY-HERE>'
```


```python
import pymongo
from superduperdb import superduper

db = pymongo.MongoClient().pymongo_docs
    
db = superduper(db)
```


```python
import json
from superduperdb.db.mongodb.query import Collection
from superduperdb.container.document import Document as D

with open('pymongo.json') as f:
    data = json.load(f)

db.execute(Collection('documents').insert_many([D(r) for r in data]))
```


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
