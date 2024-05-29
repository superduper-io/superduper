---
sidebar_position: 3
---

# SQL

`superduperdb` supports SQL databases via the [`ibis` project](https://ibis-project.org/).
With `superduperdb`, queries may be built which conform to the `ibis` API, with additional 
support for complex data-types and vector-searches.

## Inserting data

Table data must correspond to the `Schema` for that table.
Either [create a `Schema` and `Table`](../execute_api/data_encodings_and_schemas.md#create-a-table-with-a-schema)
or use [an auto-detected `Schema`](../execute_api/auto_data_types.md). Once you've 
got a `Schema`, all data inserted must conform to that `Schema`:

```python
import pandas

pandas.DataFrame([
    PIL.Image.open('image.jpg'), 'some text', 4,
    PIL.Image.open('other_image.jpg'), 'some other text', 3,
])

t.insert(dataframe.to_dict(orient='records'))
```

## Selecting data

`superduperdb` supports selecting data via the `ibis` query API.
For example:

```python
db['my_table'].filter(t.rating > 3).limit(5).select(t.image).execute()
```

### Vector-search

Vector-searches are supported via the `like` operator:

```python
(
    db['my_table']
    .like({'text': 'something like this'}, vector_index='my-index')
    .filter(t.rating > 3)
    .limit(5)
    .select(t.image, t.id)
).execute()
```

Vector-searches are either first or last in a chain of operations:

```python
(
    db['my_table']
    t.filter(t.rating > 3)
    .limit(5)
    .select(t.image, t.id)
    .like({'text': 'something like this'}, vector_index='my-index')
).execute()
```

## Updating data

Updates are not covered for `superduperdb` SQL integrations.

## Deleting data

```python
db.databackend.drop_table('my-table')
```
