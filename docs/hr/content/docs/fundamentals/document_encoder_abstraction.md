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
then the `Document` instance contains calls to `Encoder` instances.

### `Encoder`

The `Encoder` class, allows users to create and encoder custom datatypes, by providing 
their own serializers.

Here is an example of applying an `Encoder` to add an image to a `Document`:

```python
import pickle
import PIL.Image
from superduperdb import Encoder, Document

image = PIL.Image.open('my_image.jpg')

my_image_encoder = Encoder(
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

### `Schema`

A `Schema` allows developers to connect named fields of dictionaries 
or columns of `pandas.DataFrame` objects with `Encoders`. 

A `Schema` is used, in particular, for SQL databases/ tables, and for 
models that return multiple outputs.

Here is an example `Schema`, which is used together with text and image 
fields:

```python
s = Schema('my-schema', fields={'my-text': 'str', 'my-image': my_image_encoder})
```
