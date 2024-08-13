---
sidebar_position: 5
---

# Encoding data

In AI, typical types of data are:

- **Numbers** (integers, floats, etc.)
- **Text**
- **Images**
- **Audio**
- **Videos**
- **...bespoke in house data**

Most databases don't support any data other than numbers and text.
Superduper enables the use of these more interesting data-types using the `Document` wrapper.

### `Document`

The `Document` wrapper, wraps dictionaries, and is the container which is used whenever 
data is exchanged with your database. That means inputs, and queries, wrap dictionaries 
used with `Document` and also results are returned wrapped with `Document`.

Whenever the `Document` contains data which is in need of specialized serialization,
then the `Document` instance contains calls to `DataType` instances.

### `DataType`

The [`DataType` class](../apply_api/datatype), allows users to create and encoder custom datatypes, by providing 
their own encoder/decoder pairs.

Here is an example of applying an `DataType` to add an image to a `Document`:

```python
import pickle
import PIL.Image
from superduper import DataType, Document

image = PIL.Image.open('my_image.jpg')

my_image_encoder = DataType(
    identifier='my-pil',
    encoder=lambda x, info: pickle.dumps(x),
    decoder=lambda x, info: pickle.loads(x),
)
```

When all data is inserted into the database, each piece of data is encoded using the corresponding datatype. 
```
>> encoded_data = my_image_encoder.encode_data(image)
>> encoded_data
b'\x80\x04\x95[\x00\x00\x00\x00\x00\x00\x00\x8c\x12PIL.PngImagePlugin\x94\x8c\x0cPngImageFile\x94\x93\x94)\x81\x94]\x94(}\x94\x8c\x0ctransparency\x94K\x00s\x8c\x01P\x94K\x01K\x01\x86\x94]\x94(K\x00K\x00K\x00eC\x01\x00\x94eb.'
```

When the data is retrieved from the database, it is decoded accordingly.
```python
>>> my_image_encoder.decode_data(encoded_data)
<PIL.PngImagePlugin.PngImageFile image mode=P size=1x1>
```

By default, data encoded with `DataType` is saved in the database, but developers 
may alternatively save data in the `db.artifact_store` instead. 

This may be achiever by specifying the `encodable=...` parameter:

```python
my_image_encoder = DataType(
    identifier='my-pil',
    encoder=lambda x, info: pickle.dumps(x),
    decoder=lambda x, info: pickle.loads(x),
    encodable='artifact',    # saves to disk/ db.artifact_store
    # encodable='lazy_artifact', # Just in time loading
)
```

### `Schema`

A `Schema` allows developers to connect named fields of dictionaries 
or columns of `pandas.DataFrame` objects with `DataType` instances.

A `Schema` is used, in particular, for SQL databases/ tables, and for 
models that return multiple outputs.

Here is an example `Schema`, which is used together with text and image 
fields:

```python
schema = Schema('my-schema', fields={'my-text': 'str', 'my-img': my_image_encoder})
```

All data is encoded using the schema when saved, and decoded using the schema when queried.

```python
>>> saved_data = Document({'my-img': image}).encode(schema)
>>> saved_data
{'my-img': b'\x80\x04\x95[\x00\x00\x00\x00\x00\x00\x00\x8c\x12PIL.PngImagePlugin\x94\x8c\x0cPngImageFile\x94\x93\x94)\x81\x94]\x94(}\x94\x8c\x0ctransparency\x94K\x00s\x8c\x01P\x94K\x01K\x01\x86\x94]\x94(K\x00K\x00K\x00eC\x01\x00\x94eb.',
 '_schema': 'my-schema',
 '_builds': {},
 '_files': {},
 '_blobs': {}}
```

```python
>>> Document.decode(saved_data, schema=schema).unpack()
{'my-img': <PIL.PngImagePlugin.PngImageFile image mode=P size=1x1>}
```


