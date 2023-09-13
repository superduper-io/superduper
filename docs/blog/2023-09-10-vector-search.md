---
slug: welcome
title: Introducing SuperDuperDB
subtitle: Bringing AI to your datastore in Python
authors: [blythed]
tags: [AI, data-store, Python]
---

It's 2023, and unless you've been in cryo-sleep since mid-2022, you'll have heard about the explosion of powerful AI and LLMs. Leveraging these LLMs, developers are now able to connect existing data with AI using vector search.

<!--truncate-->

AI models can allow users to extract valuable information out of their existing data stores (databases, data lakes, data warehouses), in a host of different ways: semantic search, retrieval augmented generation, visual classification, conditional image generation, recommendation systems, anomaly and fraud detection, image search, multimodal search, time-series analysis and much, much more.

In the AI-powered future, a full-featured data application will have five parts:

- Backend
- Frontend
- Data store
- AI models
- Vector search

There are many different solutions for AI model computations and vector search separately, but some deep pitfalls appear when you put both together.

### A model has no natural way to talk to a datastore

The most flexible frameworks for building AI models, like PyTorch, don’t understand text or images out-of-the-box without writing custom code.

Model libraries containing precompiled and trained AI models often support text but not computer vision or audio classification. Worse, you can’t just pass a data store connection to such a library, and tell the library to use the connection to train the model: you have to write more custom code.

There is no general Python abstraction bringing self-built models like PyTorch, models imported from libraries like Scikit-Learn, and models hosted behind APIs like OpenAI, together under one roof with existing data stores: even more custom code.

The result is that developers still must perform considerable coding to connect AI models with their data stores.

### Vector databases mean data fragmentation and difficulties with data-lineage

A vector database is powerful but leaves architects and developers with questions:

- Should all data now live in the vector database?
- Should the vector database only contain vectors?

Ideally, data would stay in the primary datastore, but many datastores do not have a vector search implementation.

On the other hand, it is problematic to make the vector database the primary datastore for an application, as most vector databases lack the specialized features of classical relational databases or document stores, and offer few consistency or performance guarantees.

### Connect models and datastores with SuperDuperDB

- SuperDuperDB is a framework which wraps data stores and AI models, with minimal boilerplate code.
- Arbitrary models from open-source are applied directly to datastore queries and the outputs can be saved right back in the datastore, keeping everything in one location. Computations scale using the rich and diverse tools in the PyData ecosystem.
- SuperDuperDB allows complex data types as inputs to AI models, such as images, audio and beyond.
- SuperDuperDB can instantly make a classical database or data store vector-searchable. SuperDuperDB wraps well-known query APIs with additional commands for vector search, and takes care of organizing the results into a consistent result set for databases without a native vector-search implementation.
- SuperDuperDB can use a query to train models directly on the data store. The fully open-source SuperDuperDB environment provides a scalable and serverless developer experience for model training.

### Get started easily, and go far

SuperDuperDB is designed to make it as simple as possible to get started. For example, to connect with SuperDuperDB using MongoDB and Python, just type:

```python
from superduperdb import superduper
from pymongo import MongoClient

db = MongoClient().my_database
db = superduper(db)
```

At the same time, SuperDuperDB is ready for the full range of modern AI tools. It scales horizontally and includes a flexible approach allowing arbitrary AI frameworks to be used together, including torch, transformers, sklearn and openai.

- GitHub: https://github.com/SuperDuperDB/superduperdb
- Docs: https://docs.superduperdb.com
- Blog: https://www.superduperdb.com/blog

### The road ahead

In the weeks and months to come we’ll be:

- Adding SQL support (already close to completion)
- Building bridges to more AI frameworks, libraries, models and API services
- Creating tools to manage a SuperDuperDB deployment in production

### Contributors are welcome!

SuperDuperDB comes with the Apache 2.0 license. We would like to encourage developers interested in open-source development to contribute in our discussion forums, issue boards and by making their own pull requests. We'll see you on GitHub!

### Become a Design Partner!

We are in the process of selecting a few visionary organizations to become our design partners. The aim is to identify and implement AI applications that could bring transformative benefits to their product offerings. We're offering this at absolutely zero cost. If you would like to learn more about this opportunity please reach out to us via email: hello@superduperdb.com. 