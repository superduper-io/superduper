# Deleting data

:::note
This functionality is only supported for MongoDB `db.databackend` implementations.
For SQL databases, users should drop unwanted tables or use native clients
to delete data.
:::

Delete queries follow exactly the same [pattern as insert queries](./basic_insertion). For example:

```python
deleted_ids, jobs = db.execute(my_collection.delete_many({}))
```