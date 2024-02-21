---
sidebar_position: 4
---

# Vector-Search Using Chunked Data on MongoDB

Let's find specific text within documents using vector-search. In this
example, we show how to do vector-search. But here, we want to go one
step further. Let's search for smaller pieces of text within larger
documents. For instance, a developer may store entire documents but
wants to find specific parts or references inside those documents.

Here we will show you an example with Wikipedia dataset. Implementing
this kind of search is usually more complex, but with `superduperdb`,
it's just one extra command.

Real-life use cases for the described problem of searching for specific
text within documents using vector-search with smaller text units
include:

1.  **Legal Document Analysis:** Lawyers could store entire legal
    documents and search for specific clauses, references, or terms
    within those documents.

2.  **Scientific Research Papers:** Researchers might want to find and
    extract specific information or references within scientific papers.

3.  **Code Search in Version Control Systems:** Developers could store
    entire code files and search for specific functions, classes, or
    code snippets within those files.

4.  **Content Management Systems:** Content managers may store complete
    articles and search for specific paragraphs or keywords within those
    articles.

5.  **Customer Support Ticket Analysis:** Support teams might store
    entire support tickets and search for specific issues or resolutions
    mentioned within the tickets.

In each of these scenarios, the ability to efficiently search for and
retrieve smaller text units within larger documents can significantly
enhance data analysis and retrieval capabilities.


## Connect to datastore

First, we need to establish a connection to a MongoDB datastore via
SuperDuperDB. You can configure the `MongoDB_URI` based on your specific
setup.

Here are some examples of MongoDB URIs:

-   For testing (default connection): `mongomock://test`
-   Local MongoDB instance: `mongodb://localhost:27017`
-   MongoDB with authentication:
    `mongodb://superduper:superduper@mongodb:27017/documents`
-   MongoDB Atlas:
    `mongodb+srv://<username>:<password>@<atlas_cluster>/<database>`


``` python
import os
from superduperdb import superduper

mongodb_uri = os.getenv("MONGODB_URI", "mongomock://test")

# SuperDuperDB, now handles your MongoDB database
# It just super dupers your database 
db = superduper(mongodb_uri)
```

To demonstrate this search technique with larger text units, we'll use a Wikipedia sample. Run this command to fetch the data.

``` python
# Downloading the Wikipedia sample JSON file
!curl -O https://superduperdb-public.s3.eu-west-1.amazonaws.com/wikipedia-sample.json
```

Just like before, we insert the data using a syntax similar to
`pymongo`:

``` python
import json
from superduperdb.backends.mongodb import Collection
from superduperdb import Document as D

# Read the first 100 records from a JSON file ('wikipedia-sample.json')
with open('wikipedia-sample.json') as f:
    data = json.load(f)[:100]

# Connect to the database and insert the data into the 'wikipedia' collection. 'D(r)' converts each record 'r' into a 'Document' object before insertion
db.execute(Collection('wikipedia').insert_many([D(r) for r in data]))
```

Let's take a look at a document

``` python
# Executing a find_one query on the 'wikipedia' collection and unpacking the result
r = db.execute(Collection('wikipedia').find_one()).unpack()

# Displaying the result
r
```

To create the search functionality, we establish a straightforward model designed to break down the raw text into segments. These segments are then stored in another collection:

``` python
from superduperdb import Model

# Define a function 'splitter' to split the 'abstract' field of a document into chunks.
def splitter(r):
    # Initialize the output list with the document title
    out = [r['title']]
    # Split the 'abstract' field into chunks of 5 words
    split = r['abstract'].split(' ')
    # Iterate over the chunks and add them to the output list
    for i in range(0, len(split) - 5, 5):
        out.append(' '.join(split[i: i + 5]))
    # Filter out empty strings from the output list
    out = [x for x in out if x]
    return out

# Create a 'Model' instance named 'splitter' with the defined 'splitter' function
model = Model(
    identifier='splitter', # Identifier for the model
    object=splitter, # The function to be used as a model
    flatten=True, # Flatten the output into a single list
    model_update_kwargs={'document_embedded': False}, # Model update arguments
)

# Use the 'predict' method of the model to get predictions for the input 'r'. one=true indicates that we only want one output to check!
model.predict(r, one=True)
```

Let's utilize this model across the entire input collection:

``` python
# Use the 'predict' method of the model
model.predict(
    X='_base', # Input data used by the model 
    db=db, # Database instance (assuming 'db' is defined earlier in your code)
    select=Collection('wikipedia').find() # MongoDB query to select documents from the 'wikipedia' collection
)
```
Now let's look at the split data:

``` python
# Using the 'execute' method to execute a MongoDB query
# Finding one document in the collection '_outputs._base.splitter'
db.execute(Collection('_outputs._base.splitter').find_one())
```

We can perform a search on this data in a manner similar to the previous
example:


``` python
from superduperdb import VectorIndex, Listener
from superduperdb.ext.openai import OpenAIEmbedding

# Create an instance of the OpenAIEmbedding model with 'text-embedding-ada-002'
model = OpenAIEmbedding(identifier= 'text-embedding-ada-002', model='text-embedding-ada-002')


# Add a VectorIndex to the database
db.add(
    VectorIndex(
        identifier=f'chunked-documents', # Identifier for the VectorIndex
        indexing_listener=Listener(
            model=model,  # Embedding model used for indexing
            key='_outputs._base.splitter', # Key to access the embeddings in the database
            select=Collection('_outputs._base.splitter').find(), # MongoDB query to select documents for indexing
            predict_kwargs={'max_chunk_size': 1000}, # Additional parameters for the model's predict method like chunk size
        ),
        compatible_listener=Listener(
            model=model, # Embedding model used for compatibility checking
            key='_base', 
            select=None,  # No specific MongoDB query for Listener
            active=False, 
        )
    )
)
```

Now we can search through the split-text collection and retrieve the
full original documents, highlighting which text was found to be
relevant:

``` python
from superduperdb.backends.mongodb import Collection
from superduperdb import Document as D
from IPython.display import *

# Define the query
query = 'politics'

# Specify the shingle and main collections
shingle_collection = Collection('_outputs._base.splitter')
main_collection = Collection('wikipedia')

# Execute a search using superduperdb
result = db.execute(
    shingle_collection
        .like(D({'_base': query}), vector_index='chunked-documents', n=5)
        .find({}, {'_outputs._base.text-embedding-ada-002': 0})
)

# Display the search results
display(Markdown(f'---'))

# Iterate over the search results
for shingle in result:
    # Retrieve the original document from the main collection
    original = db.execute(main_collection.find_one({'_id': shingle['_source']}))
    
    # Display the title of the original document
    display(Markdown(f'# {original["title"]}"'))
    
    # Highlight the shingle in the abstract of the original document
    start = original['abstract'].find(shingle['_outputs']['_base']['splitter'])

    to_format = (
        original["abstract"][:start] + '**' + '<span style="color:red">' +
        shingle["_outputs"]["_base"]["splitter"].upper() + '**' + '<span style="color:black">' +
        original["abstract"][start + len(shingle["_outputs"]["_base"]["splitter"]):]
    )
    
    # Display the formatted abstract
    display(Markdown(to_format))
    display(Markdown(f'---'))
```
