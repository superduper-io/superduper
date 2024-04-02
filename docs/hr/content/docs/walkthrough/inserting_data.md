---
sidebar_position: 2
---

# Inserting data

After configuring and connecting, you're ready to insert some data.

In `superduperdb`, data may be inserted using the connection `db`, 
or using a third-parth client.

## SuperDuperDB data insertion

Here's a guide to using `db` to insert data.

### MongoDB

```python
from superduperdb.backends.mongodb import Collection
from superduperdb import Document

db.execute(
    Collection('<collection-name>')
        .insert_many([Document(record) for record in records])
)
```

The `records` may be any dictionaries supported by MongoDB, as well as dictionaries
containing items which may converted to `bytes` strings.

Other MongoDB clients may also be used for insertion. Here, one needs to explicitly 
take care of conversion of data to `bytes` wherever `Encoder` instances have been used.
For instance, using `pymongo`, one may do:

```python
from superduperdb import Document

collection = pymongo.MongoClient(uri='<your-database-uri>').my_database['<collection-name>']
collection.insert_many([
    Document(record).encode() for record in records
])

```

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

Native clients may also be used to insert data. Here, one needs to explicitly 
take care of conversion of data to `bytes` wherever `Encoder` instances have been used. 
For instance, in DuckDB, one may do:

```python
import duckdb
import pandas

my_df = pandas.DataFrame([Document(r).encode() for r in records])

duckdb.sql("INSERT INTO <table-name> SELECT * FROM my_df")
```