---
sidebar_position: 2
---

# Setting up tables and encodings

:::note
Note that for MongoDB this step is only required if 
you would like to use data-types not supported natively by MongoDB, 
such as images. For SQL this is a necessary step.
:::

`superduperdb` has flexible support for data-types. In both MongoDB and SQL databases,
one uses `superduperdb.DataType` to define one's own data-types.

## `DataType` abstraction

The `DataType` class requires two functions which allow the user to go to-and-from `bytes`.
Here is an `DataType` which encodes `numpy.ndarray` instances to `bytes`:

```python
import numpy
from superduperdb import DataType

my_array = DataType(
    'my-array',
    encoder=lambda x: memoryview(x).tobytes(),
    decode=lambda x: numpy.frombuffer(x),
)
```

Here's a more interesting `DataType` which encoders audio from `numpy.array` format to `.wav` file `bytes`:

```python
import librosa
import io
import soundfile

def decoder(x):
    buffer = io.BytesIO(x)
    return librosa.load(buffer)

def encoder(x):
    buffer = io.BytesIO()
    soundfile.write(buffer)
    return buffer.getvalue()

audio = DataType('audio', encoder=encoder, decoder=decoder)
```

It's completely open to the user how exactly the `encoder` and `decoder` arguments are set.

You may include these `DataType` instances in models, data-inserts and more. You can also directly 
register the `DataType` instances in the system, using:

```python
db.apply(my_array)
db.apply(audio)
```

To reload (for instance in another session) do:

```python
my_array_reloaded = db.load('datatype', 'my_array')
audio_reloaded = db.load('datatype', 'audio')
```

:::tip
Many of the `superduperdb` extensions come with their own pre-built `DataType` instances.
For example:

- `superduperdb.ext.pillow.pil_image`
- `superduperdb.ext.numpy.array`
- `superduperdb.ext.torch.tensor`
:::

Read more about `DataType` [here](../apply_api/datatype).

## Create a `Schema`

The `Schema` component wraps several columns of standard data or `DataType` encoded data; it 
may be used with MongoDB and SQL databases, but is only necessary for SQL.

Here is a `Schema` with three columns, one of the columns is a standard data-type "str".
The other 2 are given by the `DataType` instances defined above.

```python
from superduperdb import Schema
from superduperdb.ext.pillow import pil_image

my_schema = Schema(
    'my-schema',
    fields={'txt': 'str', 'audio': audio, 'img': pil_image}
)

# save this for later use
db.apply(my_schema)
```

### Table schemas in `SQL`

For SQL databases, one needs to have already created a schemas to work with tables in `superduperdb`. To register/ create a `Table` with a `Schema` in `superduperdb`, one uses `superduperdb.backends.ibis.Table`:

```python
from superduperdb.backends.ibis import Table

db.add(
    Table(
        'my-table',
        schema=my_schema,
    )
)
```