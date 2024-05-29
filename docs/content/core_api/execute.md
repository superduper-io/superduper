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
- Complex model predictions which include database queries (e.g. "RAG")

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

***MongoDB***

A MongoDB `find` query can be built like this:

```python
q = db['collection'].find().limit(5).skip(2)
```

***SQL***

A query with on an SQL data-backend can be built with `ibis` syntax like this:

```python
q = db['documents'].filter(t.brand == 'Nike').limit(5)
```

### Inserts

***MongoDB***

Typically insert queries wrap `Document` instances and call the `insert` method on a table or collection:

```python
q = db['documents'].insert_many([Document(r) for r in data])
```

***SQL***

The `ibis` insert is slightly different:

```python
q = db['documents'].insert([Document(r) for r in data])
```

## Executing the query


```python
results = q.execute()
```

***Multiple results***

Iterables of results are sent wrapped in a cursor

***Indiviudal results***

Individual results are sent wrapped in a `Document`

Read more about `.execute` [here](../execute_api/overview).