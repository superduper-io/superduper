---
sidebar_position: 26
---

# Component versioning

Whenever a `Component` is created (see [here](../Fundamentals/09_component_abstraction.md) for overview of `Component` classes),
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

Read more about `VectorIndex` and vector-searches [here](25_vector_search.mdx).