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

## Create your own `Model` subclasses

Developers may create their own `Model` sub-classes, and deploy these directly to `superduperdb`.
The key methods the developers need to create are:

- `predict_one`
- `predict`
- Optionally `fit`

### Minimal example with `prediction`

Here is a simple sub-class of `Model`:

```python
import dataclasses as dc
from superduperdb.components.model import Model
import typing as t

@dc.dataclass(kw_only=True)
class CustomModel(Model):
    signature: t.ClassVar[str] = '**kwargs'
    my_argument: int = 1

    def predict_one(self, x, y):
        return x + y + self.my_argument

    def predict(self, dataset):
        return [self.predict_one(**r) for r in dataset]
        return x + y
```

The addition of `signature = **kwargs` controls how the individual datapoints in the dataset 
are emitted, for consumption by the internal workings of the model

### Including datablobs which can't be converted to JSON

If your model contains large data-artifacts or non-JSON-able content, then 
these items should be labelled with [a `DataType`](../apply_api/datatype).

On saving, this will allow `superduperdb` to encode their values and save the result
in `db.artifact_store`.

Here is an example which includes a `numpy.array`:

```python
import numpy as np
from superduperdb.ext.numpy import array


@dc.dataclass(kw_only=True)
class AnotherModel(Model):
    _artifacts: t.ClassVar[t.Any] = [
        ('my_array', array)
    ]
    signature: t.ClassVar[str] = '**kwargs'
    my_argument: int = 1
    my_array: np.ndarray

    @ensure_initialized
    def predict_one(self, x, y):
        return x + y + self.my_argument + self.my_array

    def predict(self, dataset):
        return [self.predict_one(**r) for r in dataset]

my_array = numpy.random.randn(100000, 20)
my_array_type = array('my_array', shape=my_array.shape, encodable='lazy_artifact')
db.apply(my_array_type)

m = AnotherModel(
    my_argument=2,
    my_array=my_array,
    artifacts={'my_array': my_array_type},
)
```

When `db.apply` is called, `m.my_array` will be converted to `bytes` with `numpy` functionality
and a reference to these `bytes` will be saved in the `db.metadata_store`.

Notice that the `.predict_one` method is decorated with `@ensure_initialized`.
This allows `superduperdb` to load `my_array` only when needed.

In principle any `DataType` can be used to encode such an object.
