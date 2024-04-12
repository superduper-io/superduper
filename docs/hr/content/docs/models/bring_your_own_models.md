# Bring your own models

There are two ways to bring your own computations
and models to SuperDuperDB.

1. Wrap your own Python functions
2. Write your own `Model` sub-classes

## Wrap your own Python functions

This serializes a Python object or class:

```python
from superduperdb import objectmodel

@objectmodel
def my_model(x, y):
    return x + y
```

Additional arguments may be provided to the decorator from `superduperdb.components.model.ObjectModel`:

```python
@objectmodel(num_workers=4)
def my_model(x, y):
    return x + y
```

Similarly the following snippet saves the source code of a python object instead of serializing the object:

```python
from superduperdb import codemodel

@codemodel
def my_other_model(x, y):
    return x * y
```

These decorators may also be applied to `callable` classes.
If your class has important state which should be serialized with the class, 
then use `objectmodel` otherwise you can use `codemodel`:

```python
@objectmodel
class MyClass:
    ...

    def __call__(self, x):
        ...
```

## Serialization

TODO this is all old hat.

### Serializing components with SuperDuperDB

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