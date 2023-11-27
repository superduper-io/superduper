---
sidebar_position: 14
---

# Inserting images, audio, video and other special data

We discovered earlier, that an initial step in working with `superduperdb`
is to establish the data-types one wishes to work with, create `Encoder` instances for
those data-types, and potentially
`Schema` objects for SQL tables.

If these have been created, data may be inserted which use these data-types.

A previously defined `Encoder` may be used directly to insert data to the database.

## MongoDB

```python
from superduperdb import Document

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

```python

df = pandas.DataFrame([
    {
        'array': numpy.random.randn(3, 224, 224),
        'audio': librosa.load(files[i])
    } 
    for _ in range(100)
])

db.execute(table.insert(df))
```


