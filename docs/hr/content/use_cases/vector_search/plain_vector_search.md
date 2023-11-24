---
sidebar_position: 1
---
# Text

## Vector-search with SuperDuperDB

## Introduction
This notebook provides a detailed guide on performing vector search using SuperDuperDB. Vector search is a powerful technique for searching and retrieving documents based on their similarity to a query vector. In this guide, we will demonstrate how to set up SuperDuperDB for vector search and use it to search a dataset of documents.

## Prerequisites

Before diving into the implementation, ensure that you have the necessary libraries installed by running the following commands:


```python
!pip install superduperdb
!pip install ipython
```

Additionally, ensure that you have set your openai API key as an environment variable. You can uncomment the following code and add your API key:


```python
import os

#os.environ['OPENAI_API_KEY'] = 'sk-...'

if 'OPENAI_API_KEY' not in os.environ:
    raise Exception('You need to set an OpenAI key as environment variable: "export OPEN_API_KEY=sk-..."')
```

## Connect to datastore 

First, we need to establish a connection to a MongoDB datastore via SuperDuperDB. You can configure the `MongoDB_URI` based on your specific setup. 
Here are some examples of MongoDB URIs:

* For testing (default connection): `mongomock://test`
* Local MongoDB instance: `mongodb://localhost:27017`
* MongoDB with authentication: `mongodb://superduper:superduper@mongodb:27017/documents`
* MongoDB Atlas: `mongodb+srv://<username>:<password>@<atlas_cluster>/<database>`


```python
from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
import os

mongodb_uri = os.getenv("MONGODB_URI","mongomock://test")
db = superduper(mongodb_uri, artifact_store='filesystem://./data/')

doc_collection = Collection('documents')
```


```python
db.metadata
```

## Load Dataset 

We have prepared a dataset, which is the inline documentation of the pymongo API. Let's load this dataset:


```python
!curl -O https://superduperdb-public.s3.eu-west-1.amazonaws.com/pymongo.json

import json

with open('pymongo.json') as f:
    data = json.load(f)
```

As usual, we insert the data:


```python
from superduperdb import Document

db.execute(doc_collection.insert_many([Document(r) for r in data]))
```


```python
db.execute(Collection('documents').find_one())
```

## Create Vectors

In the remainder of the notebook, you can choose between using the `openai` or `sentence_transformers` libraries to perform vector search. After instantiating the model wrappers, the rest of the notebook remains identical.

For OpenAI vectors:


```python
from superduperdb.ext.openai.model import OpenAIEmbedding

model = OpenAIEmbedding(model='text-embedding-ada-002')
```

For Sentence-Transformers vectors, uncomment the following section:


```python
#import sentence_transformers
#from superduperdb import Model, vector

#model = Model(
#    identifier='all-MiniLM-L6-v2', 
#    object=sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2'),
#    encoder=vector(shape=(384,)),
#    predict_method='encode', # Specify the prediction method
#    postprocess=lambda x: x.tolist(),  # Define postprocessing function
#    batch_predict=True, # Generate predictions for a set of observations all at once 
#)
```

## Index Vectors

Now we can configure the Atlas vector-search index. This command saves and sets up a model to `listen` to a particular subfield (or the whole document) for new text, converts it on the fly to vectors, and then indexes these vectors using Atlas vector-search.


```python
from superduperdb import Listener, VectorIndex

db.add(
    VectorIndex(
        identifier=f'pymongo-docs-{model.identifier}',
        indexing_listener=Listener(
            select=doc_collection.find(),
            key='value',
            model=model,
            predict_kwargs={'max_chunk_size': 1000},
        ),
    )
)

db.show('vector_index')
```
## Perform Vector Search

Now that the index is set up, we can use it in a query. SuperDuperDB provides some syntactic sugar for the `aggregate` search pipelines, which can be helpful. It also handles all the conversion of inputs to vectors under the hood.

```python
from superduperdb import Document
from IPython.display import *

# Define the search parameters
search_term = 'Query the database'
num_results = 5

# Execute the query
result = db.execute(doc_collection
        .like(Document({'value': search_term}), vector_index=f'pymongo-docs-{model.identifier}', n=num_results)
        .find()
)

# Display a horizontal line
display(Markdown('---'))

# Iterate through the query results and display them
for r in result:
    display(Markdown(f'### `{r["parent"] + "." if r["parent"] else ""}{r["res"]}`'))
    display(Markdown(r['value']))
    display(Markdown('---'))
```
