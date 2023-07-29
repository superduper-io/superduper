# Queries

```{note}
SuperDuperDB wraps the standard MongoDB query API. It augments
these queries with support for vector-search and recall of complex data-types.
```

SuperDuperDB queries are based on the queries of the underlying datalayer, upon which the
Datalayer is based (see the [section on the `Datalayer`](datalayer)).

Unlike some Python clients, in SuperDuperDB, queries are objects, rather then methods or functions.
This allows SuperDuperDB to serialize these queries for use in diverse tasks, such as model
applications using the `Watcher` paradigm (see {ref}`watchers`, model application, and management of vector-indices).

A query is executed as follows:

```python
db = build_datalayer()
db.execute(query)
```

## MongoDB

We currently support MongoDB as the datalayer backend. As in `pymongo` all queries operate at the collection level:

```python
from superduperdb.datalayer.mongodb.query import Collection

collection = Collection(name='documents')
```

With this collection standard query types may be executed. Whereas `pymongo` returns vanilla python dictionaries, SuperDuperDB returns dictionaries wrapped as `Document` instances:


```python
>>> db.execute(collection.find_one())
Document({'_id': ObjectId('64b89e92c08139e1cedc11a4'), 'x': Encodable(x=tensor([ 0.2059,  0.5608,  ...]), encoder=Encoder(identifier='torch.float32[512]', decoder=<Artifact artifact=<superduperdb.encoders.torch.tensor.DecodeTensor object at 0x1785b5750> serializer=pickle>, encoder=<Artifact artifact=<superduperdb.encoders.torch.tensor.EncodeTensor object at 0x1786767d0> serializer=pickle>, shape=[512], version=0)), '_fold': 'train'})
```

`Documents` are also used, whenever a query involves inserting data into the database. The reason for this,
is that the data may contain complex data-types such as images (see [the section on encoders](encoders) for more detail):

```python
from superduperdb.core.document import Document as D
>>> db.execute(
    collection.insert_many([
        D({'this': f'is a test {i}'})
        for i in range(10)
    ])
)
```

SuperDuperDB also includes a composite API, enabling support for vector-search together with the query API of the datalayer: see the [section on vector-search](vectorsearch) for details.

Supported MongoDB queries:

- `find_one`
- `find`
- `aggregate`
- `update_one`
- `update_many`
- `delete_many`

## Featurization

In some AI applications, for instance in [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning), it is useful to "represent" data using some model-derived "features".
We support featurization in combination with `find` and `find_one` queries:

To do this, one chains these methods with `.featurize`, specifying the model with which one would like to featurize. In MongoDB this comes out as:

```python
cursor = db.execute(
    collection.find().featurize({'image': 'my_resnet_50'})
)
```

See the [model section](model) for information on how to compute and keep features (model outputs)
up to date.
