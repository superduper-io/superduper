---
sidebar_position: 1
---

# Vanilla Text Vector-Search on MongoDB

This guide shows how to use SuperDuperDB for vector search, a powerful technique to find similar documents. We'll cover the setup and demonstrate searching a dataset of documents. Vector search with SuperDuperDB is a useful tool in various situations:

1. **Efficient Content Search:** Forget old full-text search. Now, use vector search to quickly find what you need in your content.

2. **Visual Product Discovery:** Users discover similar products by uploading images, making shopping easier.

3. **Smart Recommendations:** Get recommendations based on both visual and text features, making user experiences better.

4. **Content Analysis:** Identify similar content and images for thorough fact-checking and reporting.

5. **RAG Chatbots:** Essential for RAG chatbots in language models, making them more effective.

These examples show how vector search makes tasks more efficient and improves user experiences in different areas.

One of the remarkable features of SuperDuperDB is its ability to pull data from any type of database, vectorize it, and perform vector searches. Here's an example.

## Prerequisites

Before diving into the implementation, ensure that you have the necessary libraries installed by running the following commands:

```bash
!pip install superduperdb
!pip install ipython openai==1.1.2
```

Additionally, ensure that you have set your OpenAI API key as an environment variable. You can uncomment the following code and add your API key:

```python
import os

#os.environ['OPENAI_API_KEY'] = 'sk-...'

if 'OPENAI_API_KEY' not in os.environ:
    raise Exception('You need to set an OpenAI key as environment variable: "export OPEN_API_KEY=sk-..."')
```

## Connect to datastore

First, we need to establish a connection to a MongoDB datastore via SuperDuperDB. You can configure the `MongoDB_URI` based on your specific setup. 
Here are some examples of MongoDB URIs:

- For testing (default connection): `mongomock://test`
- Local MongoDB instance: `mongodb://localhost:27017`
- MongoDB with authentication: `mongodb://superduper:superduper@mongodb:27017/documents`
- MongoDB Atlas: `mongodb+srv://<username>:<password>@<atlas_cluster>/<database>`

```python
from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
import os

mongodb_uri = os.getenv("MONGODB_URI","mongomock://test")

# SuperDuperDB, now handles your MongoDB database
# It just super dupers your database by initializing a SuperDuperDB datalayer instance with a MongoDB backend and filesystem-based artifact store
db = superduper(mongodb_uri, artifact_store='filesystem://./data/')

# Reference collection named 'documents'
doc_collection = Collection('documents')
```

```python
# Overall metadata information
db.metadata
```

## Load Dataset

We have prepared a dataset, which is the inline documentation of the pymongo API. Let's load this dataset:

```python
# Download the 'pymongo.json' file using curl
!curl -O https://superduperdb-public.s3.eu-west-1.amazonaws.com/pymongo.json

# Import the json module to work with JSON data
import json

# Load the content of the 'pymongo.json' file into the 'data' variable
with open('pymongo.json') as f:
    data = json.load(f)
```

As usual, we insert the data:

```python
from superduperdb import Document

# Insert multiple documents into the 'documents' collection
db.execute(doc_collection.insert_many([Document(r) for r in data]))
```

```python
# Execute a query to find one document in the 'documents' collection
result = db.execute(Collection('documents').find_one())

# Print the result
print(result)
```

## Create Vectors

In the remainder of the notebook, you can choose between using the `openai` or `sentence_transformers` libraries to perform vector search. After instantiating the model wrappers, the rest of the notebook remains identical.

For OpenAI vectors:

```python
# Import the OpenAIEmbedding model from SuperDuperDB
from superduperdb.ext.openai.model import OpenAIEmbedding

# Initialize an instance of the OpenAIEmbedding model with the 'text-embedding-ada-002' model
model = OpenAIEmbedding(identifier= 'text-embedding-ada-002', model='text-embedding-ada-002')
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

Now we can configure the Atlas vector-search index. This command saves and sets up a

 model to `listen` to a particular subfield (or the whole document) for new text, converts it on the fly to vectors, and then indexes these vectors using Atlas vector-search.

```python
from superduperdb import Listener, VectorIndex

# Add a VectorIndex to the SuperDuperDB instance
db.add(
    VectorIndex(
        # Use a dynamic identifier based on the model's identifier
        identifier=f'pymongo-docs-{model.identifier}',
        
        # Specify an indexing listener with MongoDB collection and model
        indexing_listener=Listener(
            select=doc_collection.find(),  # MongoDB collection query
            key='value',  # Key for the documents
            model=model,  # Specify the model for processing
            predict_kwargs={'max_chunk_size': 1000},  # Additional prediction arguments
        ),
    )
)

# Display the vector indexes in the SuperDuperDB instance
db.show('vector_index')
```

## Perform Vector Search

Now that the index is set up, we can use it in a query. SuperDuperDB provides some syntactic sugar for the `aggregate` search pipelines, which can be helpful. It also handles all the conversion of inputs to vectors under the hood.

```python
# Import necessary classes
from superduperdb import Document
from IPython.display import *

# Define the search parameters
search_term = 'Query the database'
num_results = 5

# Execute the query
result = db.execute(
    doc_collection
        .like(Document({'value': search_term}), vector_index=f'pymongo-docs-{model.identifier}', n=num_results)
        .find()
)

# Display a horizontal line
display(Markdown('---'))

# Iterate through the query results and display them
for r in result:
    # Display the document's parent and res values in a formatted way
    display(Markdown(f'### `{r["parent"] + "." if r["parent"] else ""}{r["res"]}`'))
    
    # Display the value of the document
    display(Markdown(r['value']))
    
    # Display a horizontal line
    display(Markdown('---'))
```
