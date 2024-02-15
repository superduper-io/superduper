---
sidebar_position: 2
---

# Inserting data

After configuring and connecting, you're ready to insert some data.

In `superduperdb`, data may be inserted using the connection `db`, 
or using a third-parth client.

## Text or Structured Data

Here's a guide to using `db` to insert text or structured data.

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

## Unstructured Data : Images , Audio , Video and special data

An initial step in working with `superduperdb`
is to establish the data-types one wishes to work with, create `Encoder` instances for
those data-types, and potentially `Schema` objects for SQL tables. See [here](./data_encodings_and_schemas.md) for 
this information.

If these have been created, data may be inserted which use these data-types, including previously defined `Encoder` instances.

### MongoDB

```python
from superduperdb import Document

my_array = db.load('encoder', 'my_array')

files = ... # list of paths to audio files

db.execute(
    Collection('my-coll').insert_many([
        Document({
            'array': my_array(numpy.random.randn(3, 224, 224)),
            'audio': audio(librosa.load(files[i]))
        })
        for _ in range(100)
    ])
)
```

### SQL

With SQL tables, it's important to acknowledge

```python
files = ... # list of paths to audio files

table = db.load('table', 'my-table')

df = pandas.DataFrame([
    {
        'array': numpy.random.randn(3, 224, 224),
        'audio': librosa.load(files[i])
    } 
    for _ in range(100)
])

db.execute(table.insert(df))
```