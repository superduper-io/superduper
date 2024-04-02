# Execute

`db.execute` is SuperDuperDB's wrapper around standard database queries:

- Inserts
- Selects
- Updates
- Deletes

As well as model predictions:

- Prediction on single data points (streaming)
- Predictions on multiple data points (batch prediction)

And also queries which consist of a combination of model computations and data operations:

- Vector-search queries
- Model predictions

Standard database queries are built using a compositional syntax similar to that found in Python database clients 
such as `pymongo` and `ibis`. The API also includes extensions of this paradigm to cover model predictions
and vector-searches.

Read more about the differences and approaches to document stores/ SQL data-backends [here](docs/data_integrations).

## Building queries/ predictions

All queries consist of a "chain" of methods executed over a base object. The base object 
can refer to a table/ collection or a model:

```python
q = base_object.method_1(*args_1, **kwargs_1).method_2(*args_2, **kwargs_2)....
```

### Selects

A MongoDB `find` query can be built like this:

```python
from superduperdb.backends.mongodb import Collection

collection = Collection('documents')
q = collection.find().limit(5).skip(2)
```

A query with an SQL table can be built with `ibis` syntax like this:

```python
from superduperdb.backends.ibis import Table

t = table('my-table')

db.apply(t)

q = t.filter(t.brand == 'Nike').limit(5)
```

### Inserts

Typically insert queries wrap `Document` instances and call the `insert` method on a table or collection:

```python
q = collection.insert_many([Document(r) for r in data])
```

The `ibis` insert is slightly different:

```python
q = t.insert([Document(r) for r in data])
```

## Executing the query


```python
db.execute(q)
```