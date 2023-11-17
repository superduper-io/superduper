---
sidebar_position: 9
---

# The component abstraction

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

A `Stack` is a way of connecting diverse and interoperating sets of functionality. See [here](../WalkThrough/28_creating_stacks_of_functionality.md) for more details.

## Activating components

Each of the components may be registered to work with `superduperdb` by passing a component instance to `db.add`.