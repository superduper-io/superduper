---
sidebar_position: 25
---

# Setting up and accessing vector-search

Vector-search refers to the task of searching through vectors 
which are created as the output of an AI model.

## Procedural API setup

With the procedural API, a `.predict` can be used to configure vector-search
if the vector-search should only be accessible to one model.

```python
from superduperdb import vector

# m is a model which outputs vectors.
# this is signified with the `vector`, an `Encoder`
m = Model(
    ...,
    encoder=vector(shape=(256,))
)

m.predict(
    X='txt',
    select=collection.find(),
    create_vector_index=True,
)
```

## Declarative API setup

With the declarative API, it's possible to create two models 
which are compatible with the vectors for performing searches:

```python
from superduperdb import Listener, VectorIndex, vector

db.add(
    VectorIndex(
        indexing_listener=Listener(
            model=model_1,    # both models output vectors
            key='key-1',
            select=collection.find(),   # portion of data to calculate vectors for
        ),
        compatible_listener=Listener(
            model=model_2,
            key='key-2',
            active=False,     # this listener doesn't compute vectors on incoming data
        )
    )
)
```

## Querying the `VectorIndex` with the hybrid query-API

SuperDuperDB supports queries via:

- `pymongo`
- `ibis`

Read more about this [here](../data_integrations/supported_query_APIs.md).

In order to use vector-search in a query, one combines these APIs with the `.like` operator.

The order of the standard parts of the query and `.like` may be permuted. This gives 
2 different algorithms:

1. Find similar items based on `txt="something like this"`
2. Filter these where these similar items have the brand `"Nike"`

...versus:

1. Find items with the brand `"Nike"`
2. Find where these items are similar to `"something like this"` based on the `"txt"` field

### PyMongo

In `pymongo` one does:

```python
from superduperdb import Document
from superduperdb.backends.mongodb import Collection

collection = Collection('mycollection')

db.execute(
    collection
        .like(Document({'txt': 'something like this'}, vector_index='my-index'))
        .find({'brand': 'Nike'}, {'txt': 1})
)
```

... or

```python
db.execute(
    collection
        .like(Document({'txt': 'something like this'}, vector_index='my-index'))
        .find({'brand': 'Nike'}, {'txt': 1})
)
```

### SQL

First you need to have set-up a table. Read how to do that [here](../data_integrations/sql.md).

```python
from superduperdb import Document

my_table = db.load('my_table', 'table')

db.execute(
    my_table
        .like(Document({'txt': 'something like this'}), index='my-index')
        .filter(my_table.brand == 'Nike')
        .limit(10)
)
```

... or

```python
db.execute(
    my_table
        .filter(my_table.brand == 'Nike')
        .limit(10)
        .like(Document({'txt': 'something like this'}), index='my-index')
)
```