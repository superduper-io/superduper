---
sidebar_position: 14
---

# Inserting images, audio, video and other special data

An initial step in working with `superduperdb`
is to establish the data-types one wishes to work with, create `Encoder` instances for
those data-types, and potentially `Schema` objects for SQL tables. See [here](./data_encodings_and_schemas.md) for 
this information.

If these have been created, data may be inserted which use these data-types, including previously defined `Encoder` instances.

## MongoDB

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

## SQL

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