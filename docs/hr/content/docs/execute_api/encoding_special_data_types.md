---
sidebar_position: 14
---

# Inserting difficult datatypes with `DataType` and/ or `Schema`

In order to insert data not supported by the `db.databackend`, developers
may use `DataType` and/ or `Schema` instances to convert their data 
to encoded `bytes` in the `db.databackend`. When data is selected, 
`superduperdb` reinterprets this data in its original form (native Python images, audio, etc..).

## MongoDB

### Direct encoding with `DataType`

In MongoDB, one wraps the item to be encoded with the `DataType`.
For example, continuing the example from [here](./data_encodings_and_schemas.md#datatype-abstraction), 
we do the following:

```python
from superduperdb.backends.mongodb import Collection
import librosa 
from superduperdb.ext.pillow import pil_image

my_images = [
    PIL.Image.open(path)
    for path in os.listdir('./') if path.endswith('.jpeg')
]
my_audio = [
    librosa.load(path)
    for path in os.listdir('./') if path.endswith('.wav')
]
with open('text.json') as f:
    my_text = json.load(f)

data = [
    Document({
        'img': pil_image(x),
        'audio': audio(y),
        'txt': z
    })
    for x, y, z in zip(my_images, my_audio, my_text)
]

db.execute(
    Collection('docs').insert_many(data)
)
```

### Use with `Schema`

First developers should [create a `Schema`](./data_encodings_and_schemas).
Then they may refer to the `Schema` in the data insert:

```python
data = [
    Document({
        'img': pil_image(x),
        'audio': audio(y),
        'txt': z
    })
    for x, y, z in zip(my_images, my_audio, my_text)
]

db.execute(
    Collection('docs').insert_many(data, schema='my-schema')
)
```

## SQL

With SQL it's necessary to first [set up a `Table` with a `Schema`](./data_encodings_and_schemas#table-schemas-in-sql).
With this table `t`, one directly inserts the data:

```python 
t.insert([
    Document({
        'img': pil_image(x),
        'audio': audio(y),
        'txt': z
    })
    for x, y, z in zip(my_images, my_audio, my_text)
])
```