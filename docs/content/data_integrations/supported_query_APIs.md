---
sidebar_position: 1
---

# Community support

In order to specify the action of models on the data, we provide an interface to pythonic ecosystem query APIs.
In particular, we provide wrappers to these projects to create database queries:

- [`pymongo`](https://pymongo.readthedocs.io/en/stable/) for MongoDB
- [`ibis`](https://ibis-project.org/) for SQL databases

`ibis` also allows users to use raw SQL in their workflows.

Queries in these two-worlds can be built by importing the table/collection class from 
each data backend. With `pymongo`, one can write:

```python
query = db['products'].find({'brand': 'Nike'}, {'_id': 1}).limit(10)
```

In `ibis`, one would write:

```python
query = db['products'].filter(products.brand == 'Nike').select('id').limit(10)
```

## Hybrid API

On top of the native features of `pymongo` and `ibis`, `superduperdb` builds several novel features:

- Additional ways to query the database with the outputs of machine learning models
  - Query model-outputs directly
  - Vector-search
- Ways to encode and query more sophisticated data-types using the `Document`-`Encoder` pattern.