# `Table`

- Create a table in an SQL database, which has a `Schema` attached
- Table can be a `MongoDB` collection or an SQL table.

***Dependencies***

- `Schema`

***Usage pattern***

(Learn how to build a `Schema` [here](schema))

```python
from superduperdb.backends.ibis import Table

table = Table(
    'my-table',
    schema=my_schema
)

db.apply(table)
```