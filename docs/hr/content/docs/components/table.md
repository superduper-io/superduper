# `Table`

***Scope***

- Create a table in an SQL database, which has a `Schema` attached
- Only relevant to SQL databases
- `Table` should be used before adding data

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

db.add(table)
```

***See also***

- [SQL query API](../query_api/sql_queries)