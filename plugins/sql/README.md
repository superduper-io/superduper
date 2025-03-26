<!-- Auto-generated content start -->
# superduper_ibis

Superduper ibis is a plugin for ibis-framework that allows you to use Superduper as a backend for your ibis queries.

This plugin cannot be used independently; it must be used together with `superduper_ibis`.


Superduper supports SQL databases via the ibis project. With superduper, queries may be built which conform to the ibis API, with additional support for complex data-types and vector-searches.


## Installation

```bash
pip install superduper_ibis
```

## API


- [Code](https://github.com/superduper-io/superduper/tree/main/plugins/ibis)
- [API-docs](/docs/api/plugins/superduper_ibis)

| Class | Description |
|---|---|
| `superduper_ibis.data_backend.IbisDataBackend` | Ibis data backend for the database. |
| `superduper_ibis.query.IbisQuery` | A query that can be executed on an Ibis database. |
| `superduper_ibis.db_helper.DBHelper` | Generic helper class for database. |
| `superduper_ibis.db_helper.ClickHouseHelper` | Helper class for ClickHouse database. |
| `superduper_ibis.field_types.FieldType` | Field type to represent the type of a field in a table. |



<!-- Auto-generated content end -->

<!-- Add your additional content below -->

## Connection examples

### MySQL

```python
from superduper import superduper

db = superduper('mysql://<mysql-uri>')
```

### Postgres

```python
from superduper import superduper

db = superduper('postgres://<postgres-uri>')
```

### Other databases

```python

from superduper import superduper

db = superduper('<database-uri>')
```

## Query examples

### Inserting data

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

### Selecting data

`superduper` supports selecting data via the `ibis` query API.
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

### Updating data

Updates are not covered for `superduper` SQL integrations.

### Deleting data

```python
db.databackend.drop_table('my-table')
```
