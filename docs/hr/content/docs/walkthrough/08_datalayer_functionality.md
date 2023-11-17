---
sidebar_position: 8
---

# Datalayer functionality

Once you have connected to your database with `superduperdb` (see [here](../WalkThrough/04_connecting.md)),
you are ready to use your `db: Datalayer` object to combine two aspects of your workflow:

- Querying, and utilizing your database through the `databackend`
- Adding and removing models and associated components
- Displaying information related to these components

## Key methods

Here are the key methods which you'll use again and again:

### `db.execute`

This method executes a query. For an overview of how this works see [here](../WalkThrough/11_supported_query_APIs.md).

### `db.add`

This method adds `Component` instances to the `db.artifact_store` connection, and registers meta-data
about those instances in the `db.metadata_store`.

In addition, each sub-class of `Component` has certain "set-up" tasks, such as inference, additional configurations, 
or training, and these are scheduled by `db.add`.

<!-- See [here]() for more information about the `Component` class and it's descendants. -->

### `db.show`

This methods displays which `Component` instances are registered with the system.

### `db.remove`

This method removes a `Component` instance from the system.

## Additional methods

### `db.validate`

Validate your components (mostly models)

### `db.predict`

Infer predictions from models hosted by `superduperdb`. Read more about this and about models [here](../WalkThrough/21_apply_models.mdx).