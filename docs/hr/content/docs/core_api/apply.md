# Apply

In SuperDuperDB there are three fundamental base components which you'll use for almost all functionality:

- `Model`
- `Listener`
- `VectorIndex`

In addition there is an overarching component:

- `Stack`

which in some sense "rules them all"

Whenever you wish to apply AI to your data, you will instantiate one of more of these, and "apply" these to 
your connection:

```python
db.apply(component)
```

## Base components

### `Model`

A `Model` is a wrapper around a standard ML/ AI model. It may contain additional functionality, such as 
pre- and post-processing, and encoding/ decoding data into/ from the correct type required by the database.

`db.apply(model)` tells SuperDuperDB to store the model and it's metadata in the system.

If additional configurations, such as training parameters, are added to the `Model` then the `db.apply` command
will also train the component on data in SuperDuperDB.

Read more about `Model` [here](../components/model).

### `Listener`

A `Listener` wraps a `Model`. The `db.apply(listener)` tells SuperDuperDB to "listen" for incoming data and to compute outputs on those data, saving them back in SuperDuperDB.

Read more about `Listener` [here](../components/listener).

### `VectorIndex`

A `VectorIndex` wraps one or two `Listener` components, and tells SuperDuperDB that the outputs computed, should
be made searchable via vector-search queries.

Read more about `VectorIndex` [here](../components/vector_index).

## Connecting component: `Stack`

A `Stack` of AI functionality is a combination of multiple `Model`, `Listener`, and `VectorIndex` components which may be "applied" in 
one go to your data via SuperDuperDB. 

On `db.add(stack)` SuperDuperDB performs the heavy lifting of deciding which components need to be applied 
first, which need to be modified on incoming data, and which outputs need to be made searchable.

Read more about `Stack` [here](../components/vector_index).