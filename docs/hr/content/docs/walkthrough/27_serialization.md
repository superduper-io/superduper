---
sidebar_position: 27
---

# Serializing components with SuperDuperDB

When adding a component to `SuperDuperDB`, 
objects which cannot be serialized to `JSON` 
are serialized to `bytes` using one of the inbuilt
serializers:

- `pickle`
- `dill`
- `torch`

Users also have the choice to create their own serializer, 
by providing a pair of functions to the `Component` descendant
`superduperdb.Serializer`.

Here is an example of how to do that, with an example `tensorflow.keras` model, 
which isn't yet natively supported by `superduperdb`, but 
may nevertheless be supported using a custom serializer:

```python
from superduperdb import Serializer

from tensorflow.keras import Sequential, load
from tensorflow.keras.layers import Dense

model = Sequential([Dense(1, input_dim=1, activation='linear')])


def encode(x):
    id = uuid.uuid4()
    x.save(f'/tmp/{id}')
    with open(f'/tmp/{id}', 'rb') as f:
        b = f.read()
    os.remove(f'/tmp/{id}')
    return b


def decode(x)
    id = uuid.uuid4()
    with open(f'/tmp/{id}', 'wb') as f:
        f.write(x)
    model = load(f'/tmp/{id}')
    os.remove(f'/tmp/{id}')
    return model


db.add(
    Serializer(
        'keras-serializer',
        encoder=encoder,
        decoder=decoder,
    )
)

db.add(
    Model(
        'my-keras-model',
        object=model,
        predict_method='predict',
        serializer='keras-serializer',
    )
)
```