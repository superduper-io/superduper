---
slug: rag-question-answering
title: Building a documentation chatbot using FastAPI, React, MongoDB and SuperDuperDB
authors: [nenb]
tags: [RAG, vector-search]
---

Imagine effortlessly infusing AI into your data repositories—databases, data warehouses, or data lakes—without breaking a sweat. With SuperDuperDB, we aim to make this dream a reality. We want to provide everyone with the tools to build AI applications directly on top of their data stores, with just a pinch of Python magic sprinkled on top! 🐍✨

In this latest blog post we take a dive into one such example - a Retrieval Augmented Generation (RAG) app we built directly on top of our MongoDB store (pretty quickly).

Since we’re in the business of building open-source software, a logical in-house application of our own technology is a question-answering app, directly on our own documentation. We built this app using SuperDuperDB together with FastAPI, React and MongoDB (the “FARMS” stack).

We use retrieval augmented generation, or RAG, to integrate an existing Large Language Model (LLM) with our own data; including documents found in vector-search in an initial pass, enables using an LLM on a domain it was not trained on. SuperDuperDB allows developers to apply RAG to their own standard database, instead of insisting that users migrate a portion of their data to a vector-search database such as Pinecone, Chroma or Milvus.

Although SuperDuperDB’s functionality is more general than simply RAG and vector search, if a model’s output does indeed consist of vectors, it’s dead easy with SuperDuperDB to use these vectors downstream in vector search and RAG applications. We’ll post more about the range of possibilities with SuperDuperDB in the coming weeks.

** 🤖 Let's ask the chatbot to tell us more about SuperDuperDB.**

:::info
Q. What is SuperDuperDB?
:::

*SuperDuperDB is a Python package that provides tools for developers to apply AI and machine learning in their already deployed datastore. It also facilitates the setup of a scalable, open-source, and auditable environment for AI development. SuperDuperDB aims to make the integration of AI and data easier, extensible, and comprehensive. It allows for easy evaluation of model predictions and insertion back into the datastore and enables training deployment with a simple one-line command.*

### Choosing our stack

Right out of the box, SuperDuperDB supports MongoDB, a popular NoSQL database among full-stack developers. MongoDB's cloud service also provides a generous free-tier offering, and and we chose this for our storage.

** 🚧 SuperDuperDB has experimental support for SQL databases which will be greatly expanded in the coming weeks and months!**

We chose FastAPI for the web framework because it creates a self-documenting server, it’s extremely full-featured, and has a large community of users - and yes, because it’s trendy. The FARM stack combines both MongoDB and FastAPI, and so it seemed natural to build our RAG app by adding SuperDuperDB to FARM to make FARMS!

### Setting up the code

We decided to stick fairly closely to a typical FastAPI directory structure, the major difference being that we now have a new ai/ subdirectory that contains two new modules: artifacts.py and components.py.

```
backend
|___ ai
|   |___ __init__.py
│   |___ artifacts.py
|   |___ components.py
│   |___ utils      # AI helper functions here
│        |__ ...
|___ documents      # Our REST backend has a single 'documents' route
|   |___ __init__.py
|   |___ models.py  # Pydantic models here
|   |___ routes.py  # AI-enhanced CRUD logic here
|___ __init__.py
|___ app.py
|___ config.py
|___ main.py
```

### Artifacts

** 🤖 Let's Question The Docs to learn more about Artifacts.**

:::info
Q. What is an Artifact? Give an example.
:::

An Artifact carries the necessary information and serialization method to save an object in a configured artifact store. The Artifact ensures that an object can be serialized and stored in a permanent and centralized location for future retrieval. An example of an artifact could be an image file named my_image.jpg that is saved in a configured artifact store, such as gridfs on MongoDB.

To build a program, you first must understand its data, and a RAG app is no different. Here, our data source are Markdown files, and we want to process them in a way which is most suitable for answering the questions we would like the LLM to answer. Here there's a trade-off: splitting the text into too large chunks, makes it harder to get good specificity in the vector-search step of RAG. However, splitting the docs into larger chunks, allows the LLM to use coherently ordered text to contextualize the answer it formulates.

SuperDuperDB supports a wide range of models for prediction and training, and flexible serialization: for instance, we might use spacy for pre-processing labels, torchvision for vectorizing images and transformers for multi-modal retrieval. (But the program is not dependent on all these models! “Don’t pay for what you don’t use” is our motto.)

Once we have our artifacts, superduperdb takes care of the rest. All serialization, creation and tracking of metadata, and job orchestration is handled automatically: the ultimate goal is to make the development of AI applications possible for anyone. For our RAG app, this step looks roughly like the following:

```python
from superduperdb.container.document import Document
from superduperdb.db.mongodb.query import Collection
...

# `artifacts` are chunked Markdown files
documents = [Document({"KEY": v}) for v in artifacts]
db.execute(Collection(name="NAME").insert_many(documents))
```

### Components

** 🤖 QtD again!**

:::info
Q. Give a short summary of Components and provide examples.
:::

*SuperDuperDB has several components that work together to provide a unified user experience for the programmer:*

- *DB: This component handles the underlying datastore and provides the functionality for storing and retrieving data. It is the core component of SuperDuperDB.*

- *Models: The Models component defines the structure and behavior of the data to be stored in the DB. It includes classes or structures that represent different types of data and their relationships.*

- *Encoders: Encoders are responsible for encoding and decoding data between different formats. They allow data to be transformed into a compatible format for storage in the DB.*

- *Listeners: Listeners are event-based components that listen for specific events or triggers. They provide a way to execute code or perform actions when certain events occur in the system.*

- *Metrics: Metrics components track and measure various statistics and performance indicators of the SuperDuperDB cluster. They provide valuable insights into the system's performance and can be used for troubleshooting and optimization.*

- *Vector-Indexes: Vector-Indexes are specialized indexes that handle vector-based and similarity search operations. They improve the efficiency of searching and retrieving similar data in the DB.*

- *Datasets: Datasets are collections of data that can be used for testing, training, or any other purpose. They provide a way to manage and organize data within the SuperDuperDB cluster.*

We are only going to use that first feature, and install our AI models inside our database.

** 💡 But Components can also listen for specific events before performing an action, track statistics of database artifacts over time and even train models.**

We chose text-embedding-ada-002 for our text embedding model, which we compute on the app's own server. For the chatbot, we selected the well-known gpt-3.5-turbo; now we can start talking to our chatbot!

### Querying the database

Our app is a particularly simple example of a CRUD app without the UPDATE or DELETE actions: once we have created our artifacts, we just have to READ the database for the text most similar to our question.

### Building queries

Using SuperDuperDB to build a query to search for relevant text snippets is very similar to using a standard MongoDB driver such as pymongo, but with additional keyword arguments like n in this example, which says how many similar items to return from the database.

```python
from superduperdb.db.mongodb.query import Collection            

context = (
    Collection(name="NAME")
    .like(
        {"KEY": query},  # Example: 'What is SuperDuperDB?'
        n=5,
        vector_index="NAME2",
    )
    .find()
)
```


### Dispatching QA queries

Under the hood, SuperDuperDB can be configured to perform searches and comparisons using a vector database like the open-source LanceDB, or MongoDB Atlas, which is what we used in QtD.

Executing a query is also very similar to a standard CRUD application, except that the database needs to be wrapped in a SuperDuperDB after it is created:

```python
from pymongo import MongoClient

from superduperdb import superduper

MONGO_URI = ...

mongo_db = MongoClient(MONGO_URI)
db = superduper(mongo_db)  # It's now a super duper database!

db.execute(context)  # proceed as normal
```


### Summary

The FARM stack works well with SuperDuperDB in Python to build RAG applications

SuperDuperDB’s support for vector search allows developers to minimize the problems with LLM hallucinations, as well as extending LLM coverage to domains the LLM was not trained on

Many RAG and question-answering applications use langchain, but SuperDuperDB stands out with its lightweight third-party integrations, support for “bring your own model”, and greater scope, incorporating scalable inference and training directly in your database.

### SuperDuperDB into the future!

** 🤖 Let's see if the bot can help us decide what to do next.**

:::info
Q. What is SuperDuperDB's mission?
:::

SuperDuperDB's mission is to facilitate and accelerate the developer journey between data and AI models by creating an easy-to-use, extensible, and comprehensive Python framework for integrating AI and ML directly to the datastore. They aim to empower developers, data scientists, and architects to leverage the open-source ecosystem in their datastore deployments. They also aim to enable scalability and industrial scale deployment, provide easy-to-use tools for individual developers, allow continued use of already existing or deployed datastores without data migration, follow a fully open-source approach, and enable individuals and organizations to circumvent vendor lock-in strategies.

Thanks for reading! If you have any questions about this article, or SuperDuperDB in general, please don’t hesitate to contact us at opensource@superduperdb.com.

### Useful links

- Question the docs online!
- SuperDuperDB on GitHub
- Play with the code on GitHub