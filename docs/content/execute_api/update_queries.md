# Updating data

:::note
This functionality is only supported for MongoDB `db.databackend` implementations
:::

Update queries follow exactly the same [pattern as insert queries](./basic_insertion). For example:

```python
updated_ids, jobs = db.execute(my_collection.update_many({}, {'$set': {'brand': 'Adidas'}}))
```