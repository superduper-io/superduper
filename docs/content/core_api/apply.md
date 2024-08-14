# Apply

In superduper there are three fundamental base components which you'll use for almost all functionality:

- [`Model`](../apply_api/model)
- [`Listener`](../apply_api/listener)
- [`VectorIndex`](../apply_api/vector_index)

In addition there is an overarching component:

- [`Stack`](../apply_api/stack)

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

`db.apply(model)` tells `superduper` to store the model and its metadata in the system.

If additional configurations, such as training parameters, are added to the `Model` then the `db.apply` command
will also train the component on data in superduper.

Read more about `Model` [here](../apply_api/model).

### `Listener`

A `Listener` wraps a `Model`. The `db.apply(listener)` tells `superduper` to "listen" for incoming data and to compute outputs on those data, saving them back in `superduper`.

Read more about `Listener` [here](../apply_api/listener).

### `VectorIndex`

A `VectorIndex` wraps one or two `Listener` components, and tells `superduper` that the outputs computed, should
be made searchable via vector-search queries.

Read more about `VectorIndex` [here](../apply_api/vector_index).

## Connecting component: `Stack`

A `Stack` of AI functionality is a combination of multiple `Model`, `Listener`, and `VectorIndex` components which may be "applied" in 
one pass to your data via superduper. 

On `db.add(stack)` superduper performs the heavy lifting of deciding which components need to be applied 
first, which need to be modified on incoming data, and which outputs need to be made searchable.

Read more about `Stack` [here](../apply_api/stack).

## View applied components

Use `db.show` to view components.

View all components:

```python
>>> db.show()
[
  {'type_id': 'model', 'identifier': 'my-model'},
  {'type_id': 'model', 'identifier': 'my-other-model'}
]
```

View all components of a certain type:

```python
>>> db.show('<type_id>')
['my-model', 'my-other-model']
```

View all versions of a particular component:

```python
>>> db.show('<type_id>', '<component_identifier>')
[0, 1, 2, 3]
```

## Reloading applied components

When components are applied with `db.apply(component)`, the component is provided with a version, which may be optionally used to reload the component.
By default the latest version is reloaded:

```python
reloaded = db.load('<type_id>', '<component_identifier>')
```

```python
reloaded = db.load('<type_id>', '<component_identifier>', <version>)
```

For example to reload a model, identified by 'my-model', the first version added:

```python
reloaded_model = db.load('model', 'my-model', 0)
```

## Read more

Read more about the "apply" API [here](../apply_api/component.md).