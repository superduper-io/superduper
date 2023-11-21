# Building Q&A Assistant Using Mongo and OpenAI

## Introduction

This notebook is designed to demonstrate how to implement a document Question-and-Answer (Q&A) task using SuperDuperDB in conjunction with OpenAI and MongoDB. It provides a step-by-step guide and explanation of each component involved in the process.


## Prerequisites

Before diving into the implementation, ensure that you have the necessary libraries installed by running the following commands:


```python
!pip install superduperdb
!pip install ipython openai==0.27.6
```

Additionally, ensure that you have set your openai API key as an environment variable. You can uncomment the following code and add your API key:


```python
import os

#os.environ['OPENAI_API_KEY'] = 'sk-...'

if 'OPENAI_API_KEY' not in os.environ:
    raise Exception('Environment variable "OPENAI_API_KEY" not set')
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
db = superduper(mongodb_uri)

collection = Collection('questiondocs')
```


```python
db.metadata
```

## Load Dataset 

In this example we use the internal textual data from the `superduperdb` project's API documentation. The goal is to create a chatbot that can provide information about the project. You can either load the data from your local project or use the provided data. 

If you have the SuperDuperDB project locally and want to load the latest version of the API, uncomment the following cell:


```python
# import glob

# ROOT = '../docs/hr/content/docs/'

# STRIDE = 3       # stride in numbers of lines
# WINDOW = 25       # length of window in numbers of lines

# files = sorted(glob.glob(f'{ROOT}/*.md') + glob.glob(f'{ROOT}/*.mdx'))

# content = sum([open(file).read().split('\n') for file in files], [])
# chunks = ['\n'.join(content[i: i + WINDOW]) for i in range(0, len(content), STRIDE)]
```

Otherwise, you can load the data from an external source. The chunks of text contain code snippets and explanations, which will be used to build the document Q&A chatbot. 


```python
from IPython.display import *

Markdown(chunks[20])
```


```python
!curl -O https://superduperdb-public.s3.eu-west-1.amazonaws.com/superduperdb_docs.json

import json
from IPython.display import Markdown

with open('superduperdb_docs.json') as f:
    chunks = json.load(f)
```

You can see that the chunks of text contain bits of code, and explanations, 
which can become useful in building a document Q&A chatbot.

As usual we insert the data. The `Document` wrapper allows `superduperdb` to handle records with special data types such as images,
video, and custom data-types.


```python
from superduperdb import Document

db.execute(collection.insert_many([Document({'txt': chunk}) for chunk in chunks]))
```

## Create a Vector-Search Index

To enable question-answering over your documents, we need to setup a standard `superduperdb` vector-search index using `openai` (although there are many options
here: `torch`, `sentence_transformers`, `transformers`, ...)

A `Model` is a wrapper around a self-built or ecosystem model, such as `torch`, `transformers`, `openai`.


```python
from superduperdb.ext.openai import OpenAIEmbedding

model = OpenAIEmbedding(model='text-embedding-ada-002')
```


```python
model.predict('This is a test', one=True)
```

A `Listener` "deploys" a `Model` to "listen" to incoming data, and compute outputs, which are saved in the database, via `db`.


```python
from superduperdb import Listener

listener = Listener(model=model, key='txt', select=collection.find())
```

A `VectorIndex` wraps a `Listener`, making its outputs searchable.


```python
from superduperdb import VectorIndex

db.add(
    VectorIndex(identifier='my-index', indexing_listener=listener)
)
```


```python
db.execute(collection.find_one())
```


```python
from superduperdb.backends.mongodb import Collection
from superduperdb import Document as D
from IPython.display import *

query = 'Code snippet how to create a `VectorIndex` with a torchvision model'

result = db.execute(
    collection
        .like(D({'txt': query}), vector_index='my-index', n=5)
        .find()
)

display(Markdown('---'))

for r in result:
    display(Markdown(r['txt']))
    display(Markdown('---'))
```

## Create a Chat-Completion Component

In this step, a chat-completion component is created and added to the system. This component is essential for the Q&A functionality:


```python
from superduperdb.ext.openai import OpenAIChatCompletion

chat = OpenAIChatCompletion(
    model='gpt-3.5-turbo',
    prompt=(
        'Use the following description and code-snippets aboout SuperDuperDB to answer this question about SuperDuperDB\n'
        'Do not use any other information you might have learned about other python packages\n'
        'Only base your answer on the code-snippets retrieved\n'
        '{context}\n\n'
        'Here\'s the question:\n'
    ),
)

db.add(chat)

print(db.show('model'))
```

## Ask Questions to Your Docs

Finally, you can ask questions about the documents. You can target specific queries and use the power of MongoDB for vector-search and filtering rules. Here's an example of asking a question:


```python
from superduperdb import Document
from IPython.display import Markdown

# Define the search parameters
search_term = 'Can you give me a code-snippet to set up a `VectorIndex`?'
num_results = 5

output, context = db.predict(
    model_name='gpt-3.5-turbo',
    input=search_term,
    context_select=(
        collection
            .like(Document({'txt': search_term}), vector_index='my-index', n=num_results)
            .find()
    ),
    context_key='txt',
)

Markdown(output.content)
```

Reset the demo


```python
db.remove('vector_index', 'my-index', force=True)
db.remove('listener', 'text-embedding-ada-002/txt', force=True)
db.remove('model', 'text-embedding-ada-002', force=True)
```
