---
sidebar_position: 4
---

# Components

A **`Component`** is an object which is a combination of `JSON`-able parameters, and classes which are not 
serializable by `JSON`, but are saved as `bytes`-blobs.

## Types of component

There are several key classes of objects in `superduperdb` all of which inherit from `Component`.
All of these objects are to be found in `superduperdb.component.*`

These are:

### `Encoder`

An `Encoder` is a class which is responsible for converting special data-types in `bytes` and back.

### `Model`

A `Model` is a class which wraps a classical AI-model, but in-addition, brings additional tooling, such as the functions required 
for pre- and post-processing, as well as which `Encoder` is needed to encode outputs.

### `Listener`

A `Listener` is a class which "deploys" a `Model` to "listen" for incoming data, and calculate predictions on this data, which 
are saved back to the database.

### `VectorIndex`

A `VectorIndex` is, informally, the necessary combination of `Component` instances, to create **vector-search** functionality, end-2-end.
Formally, a `VectorIndex` is a combination of one or more `Listener` instances which have the additional property that one of them has 
vector outputs. The `Model` instances are used to create vectors for incoming queries, on the fly, as well as preparing incoming data, and
saving vectors for this data in the database.

### `Serializer`

Some `Component` objects require special serialization protocols, in order to become saveable in the `superduperdb` world.
`Serializer` is a "meta"-component which can save these custom serialization protocols for use by other `Component` instances.

### `Schema`

A `Schema` connects traditional datatypes and `Encoder` instances, with tabular data.

### `Dataset`

A `Dataset` is an immutable snapshot of the database. This immutability is important for model validation, and reproducibility,
among other tasks.

### `Metric`

A `Metric` serves the purpose of evaluating the quality of `Component` instances - in particular, `Model`, `Listener` and `VectorIndex` 
instances.

### `Stack`

A `Stack` is a way of connecting diverse and interoperating sets of functionality. See [here](../walkthrough/creating_stacks_of_functionality) for more details.

## Activating components

Each of the components may be registered to work with `superduperdb` by passing a component instance to `db.add`.

# Component versioning

Whenever a `Component` is created (see [here](../fundamentals/component_abstraction.md) for overview of `Component` classes),
information about that `Component` is saved in the `db.metadata` store.

All components come with attributes `.identifier` which is a unique identifying string for that `Component` instance.
Whenever a `Component` with the same `.identifier` is created, the old version is deprecated and a new version is created.

Which `.identifier` values are present may be viewed, for instance, for `Model` instances, with:

```python
db.show('model')
```

To view the versions of a particular model, you may do:

```python
db.show('model', '<model-identifier>')
```

To load a component of a particular type, from the `db.artifact_store` do:

```python
m = db.load('model', '<model-identifier>')
```

By default, this loads the latest version of the model into memory.

To load a particular model version, do:

```python
m = db.load('model', '<model-identifier>', version=2)
```

This works for other components, such as `VectorIndex`.

`VectorIndex` instances also contain instances of:

- `Listener`
- `Model`

When one adds the `VectorIndex` with `db.add(vector_index)`, 
the sub-components are also versioned, if a version has not already 
been assigned to those components in the same session.

Read more about `VectorIndex` and vector-searches [here](../walkthrough/vector_search.md).
