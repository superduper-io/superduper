# MongoDB Atlas vector-search with SuperDuperDB


```python
!pip install superduperdb
!pip install sentence_transformers
```

Set your `openai` key if it's not already in your `.env` variables


```python
import os

if 'OPENAI_API_KEY' not in os.environ:
    raise Exception('You need to set an OpenAI key as environment variable: "export OPEN_API_KEY=sk-..."')
```

This line allows `superduperdb` to connect to MongoDB. Under the hood, `superduperdb` sets up configurations
for where to store:
- models
- outputs
- metadata
In addition `superduperdb` configures how vector-search is to be performed.


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

We've prepared some data - it's the inline documentation of the `pymongo` API!


```python
!curl -O https://superduperdb-public.s3.eu-west-1.amazonaws.com/pymongo.json
```

We can insert this data to MongoDB using the `superduperdb` API, which supports `pymongo` commands.


```python
import json
from superduperdb.db.mongodb.query import Collection
from superduperdb.container.document import Document as D

with open('pymongo.json') as f:
    data = json.load(f)
```


```python
data[0]
```


```python
db.execute(
    Collection('documents').insert_many([D(r) for r in data])
)
```

In the remainder of the notebook you can choose between using `openai` or `sentence_transformers` to 
perform vector-search. After instantiating the model wrappers, the rest of the notebook is identical.


```python
from superduperdb.ext.openai.model import OpenAIEmbedding

model = OpenAIEmbedding(model='text-embedding-ada-002')
```


```python
import sentence_transformers
from superduperdb.container.model import Model
from superduperdb.ext.vector.encoder import vector

model = Model(
    identifier='all-MiniLM-L6-v2',
    object=sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2'),
    encoder=vector(shape=(384,)),
    predict_method='encode',
    postprocess=lambda x: x.tolist(),
    batch_predict=True,
)
```


```python
model.predict('This is a test', one=True)
```

Now we can configure the Atlas vector-search index. 
This command saves and sets up a model to "listen" to a particular subfield (or whole document) for
new text, and convert this on the fly to vectors which are then indexed by Atlas vector-search.


```python
from superduperdb.container.vector_index import VectorIndex
from superduperdb.container.listener import Listener

db.add(
    VectorIndex(
        identifier='pymongo-docs',
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
db.show('vector_index')
```

Now the index is set up we can use it in a query. `superduperdb` provides some syntactic sugar for 
the `aggregate` search pipelines, which can trip developers up. It also handles 
all conversion of inputs to vectors under the hood


```python
from superduperdb.db.mongodb.query import Collection
from superduperdb.container.document import Document as D
from IPython.display import *

query = 'Query the database'

result = db.execute(
    Collection('documents')
        .like(D({'value': query}), vector_index='pymongo-docs', n=5)
        .find()
)

display(Markdown('---'))

for r in result:
    display(Markdown(f'### `{r["parent"] + "." if r["parent"] else ""}{r["res"]}`'))
    display(Markdown(r['value']))
    display(Markdown('---'))
```
