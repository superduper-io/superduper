---
sidebar_position: 2
---

# Inserting data

After configuring and connecting, you're ready to insert some data.

In SuperDuperDB, data may be inserted using the SuperDuperDB connection `db`, 
or using a third-parth client.

## SuperDuperDB data insertion

Here's a guide to using `db` to insert data.

### MongoDB

```python
from superduperdb.backends.mongodb import Collection

db.execute(
    Collection('<collection-name>')
        .insert_many([Document(record) for record in records])
)
```

The `records` may be any dictionaries supported by MongoDB, as well as dictionaries
containing items which may converted to `bytes` strings.

### SQL

Similarly

```python
from superduperdb.backends.ibis import Table

db.execute(
    Table('<table-name>')
        .insert([Document(record) for record in records])
)
```

The `records` must conform in their keys to the columns set by the `Schema` set.

Similarly, you may also use `pandas` dataframes:

```python
from superduperdb.backends.ibis import Table
import pandas

db.execute(
    Table('<table-name>')
        .insert(pandas.DataFrame(records))
)
```