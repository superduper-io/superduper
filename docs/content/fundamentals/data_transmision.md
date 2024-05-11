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
SuperDuperDB enables the use of these more interesting data-types using the `Document` wrapper.

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
from superduperdb import DataType, Document

image = PIL.Image.open('my_image.jpg')

my_image_encoder = DataType(
    identifier='my-pil',
    encoder=lambda x: pickle.dumps(x),
    decoder=lambda x: pickle.loads(x),
)

document = Document({'img': my_image_encoder(image)})
```

The bare-bones dictionary may be exposed with `.unpack()`:

```python
>>> document.unpack()
{'img': <PIL.PngImagePlugin.PngImageFile image mode=P size=400x300>}
```

By default, data encoded with `DataType` is saved in the database, but developers 
may alternatively save data in the `db.artifact_store` instead. 

This may be achiever by specifying the `encodable=...` parameter:

```python
my_image_encoder = DataType(
    identifier='my-pil',
    encoder=lambda x: pickle.dumps(x),
    decoder=lambda x: pickle.loads(x),
    encodable='artifact',    # saves to disk/ db.artifact_store
    # encodable='lazy_artifact', # Just in time loading
)
```

The `encodable` specifies the type of the output of the `__call__` method, 
which will be a subclass of `superduperdb.components.datatype._BaseEncodable`.
These encodables become leaves in the tree defines by a `Document`.

### `Schema`

A `Schema` allows developers to connect named fields of dictionaries 
or columns of `pandas.DataFrame` objects with `DataType` instances.

A `Schema` is used, in particular, for SQL databases/ tables, and for 
models that return multiple outputs.

Here is an example `Schema`, which is used together with text and image 
fields:

```python
s = Schema('my-schema', fields={'my-text': 'str', 'my-image': my_image_encoder})
```
