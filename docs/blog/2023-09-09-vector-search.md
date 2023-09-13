---
slug: connect-your-traditional-database-with-vector-search-using-superduperdb
title: Connect your Traditional Database with Vector-Search using SuperDuperDB
authors: [blythed]
tags: [AI, vector-search]
---

In 2023 vector-databases are hugely popular; they provide the opportunity for developers to connect LLMs, such as OpenAI’s GPT models, with their data, as well as providing the key to deploying “search-by-meaning” on troves of documents.

However: a key unanswered question, for which there is no widely accepted answer, is:

How do the vectors in my vector-database get there in the first place?

<!--truncate-->

Vectors (arrays of numbers used in vector-search) differ from the content of most databases, since they need to be calculated on the basis of other data.

Currently there are 2 approaches:

** Possibility 1: models live together with the database to create vectors at insertion time **

When data is inserted into a vector-database, the database may be configured to “calculate” or “compute” vectors on the basis of this data (generally text). This means that the database environment also has access to some compute and AI models, or access to APIs such as OpenAI, in order to obtain vectors.

Examples of this approach are:

- Weaviate (support for a range of pre-defined models, some support for bringing own model)
- Chroma (support for OpenAI and sentence_transformers)

Pros:

- The data and compute live together, so developers don’t need to create an additional app in order to use the vector-database

Cons:

- Developers are limited by the models available in the vector-database and the compute resources on the vector-database server
- Primary data needs to be stored in the vector-database; classic-database + external vector-search isn’t an expected pattern.
- Training of models is generally not supported.

** Possibility 2: the vector-database requires developers to provide their own vectors with their own models **

In this approach, developers are required to build an app which deploys model computations over data which is extracted from the datastore.

Examples of this approach are:

- LanceDB
- Milvus

Pros:

- By developing a vector-computation app, the user can use the full flexibility of the open-source landscape for computing these vectors, and can architect compute resources independently from vector-database resources
- The vector-database “specializes” in vector-search and storage of vectors, giving better performance guarantees as a result

Cons:

- Huge overhead of building one’s own computation app.
- All communication between app, vector-database and datastore (if using external datastore) must be managed by the developer

### Enter SuperDuperDB

SuperDuperDB is a middle path to scalability, flexiblity and ease-of-use in vector-search and far beyond.

- SuperDuperDB is an open-source Python environment which wraps databases and AI models with additional functionality to make them “ready” to interface with one-another; developers are able to host their data in a “classical” database, but use this database as a vector-database.
- SuperDuperDB allows users to integrate any model from the Python open source ecosystem (torch, sklearn, transformers, sentence_transformers as well as OpenAI’s API), with their datastore. It uses a flexible scheme, allowing new frameworks and code-bases to be integrated without requiring the developer to add additional classes or functionality.
- SuperDuperDB can be co-located with the database in infrastructure, but at the same time has access to its own compute, which is scalable. This makes it vertically performant and at the same time, ready to scale horizontally to accommodate larger usage.
- SuperDuperDB enables training directly with the datastore: developers are only required to specify a database-query to initiate training on scalable compute.
- Developers are not required to program tricky boilerplate code or architectures for computing vector outputs and storing these back in the database. This is all supported natively by SuperDuperDB.
- SuperDuperDB supports data of arbitrary type: with its flexible serialization model, SuperDuperDB can handle text, images, tensors, audio and beyond.
- SuperDuperDB’s scope goes far beyond vector-search; it supports models with arbitrary outputs: classification, generative AI, fore-casting and much more are all within scope and supported. This allows users to build interdependent models, where base models feed their outputs into downstream models; this enables transfer learning, and quality assurance via classification on generated outputs, to name but 2 key outcomes of SuperDuperDB’s integration model.

### Minimal boilerplate to connect to SuperDuperDB

Connecting to MongoDB with SuperDuperDB is super easy. The connection may be used to insert documents, although insertion/ ingestion can also proceed via other sources/ client libraries.

```python
import json
import pymongo

from superduperdb import superduper
from superduperdb.container.document import Document
from superduperdb.db.mongodb.query import Collection

db = pymongo.MongoClient().documents
db = superduper(db)

collection = Collection('wikipedia')

with open('wikipedia.json') as f:
    data = json.load(f)

db.execute(
    collection.insert_many([Document(r) for r in data])
)
```

### Set up vector-search with SuperDuperDB in one command

```python
from superduperdb.container.vector_index import VectorIndex
from superduperdb.container.listener import Listener
from superduperdb.ext.numpy.array import array
from superduperdb.ext.openai import OpenAIEmbedding

db.add(
    VectorIndex(
        identifier=f'wiki-index',
        indexing_listener=Listener(
            model=OpenAIEmbedding(model='text-embedding-ada-002'),
            key='abstract',
            select=collection.find(),
            predict_kwargs={'max_chunk_size': 1000},
        )
    )
)
```

This approach is simple enough to allow models from a vast range of libraries and sources to be implemented: open/ closed source, self-built or library based and much more.

Now that the index has been created, queries may be dispatched in a new session to SuperDuperDB without reloading the model:

```python
cur = db.execute(
    collection
        .like({'title': 'articles about sport'}, n=10, vector_index=f'wiki-index')
        .find({}, {'title': 1})
)

for r in cur:
    print(r)
```

The great thing about using MongoDB or a similar battle tested database for vector-search, is that it can be easily combined with important filtering approaches. In this query, we restrict the results to a hard match involving the word “Australia”:

```python
cur = db.execute(
    collection
        .like({'title': 'articles about sport'}, n=100, vector_index=f'wiki-index-{model.identifier}')
        .find({'title': {'$regex': '.*Australia'}})
)

for r in cur:
    print(r['title'])
```

### SuperDuperDB is licensed under Apache 2.0 and is a community effort!

We would like to encourage developers interested in the project to contribute in our discussion forums, issue boards and by making their own pull requests. We'll see you on GitHub!