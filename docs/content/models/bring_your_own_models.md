# Bring your own models

There are two ways to bring your own computations
and models to Superduper.

1. Wrap your own Python functions
2. Write your own `Model` sub-classes

## Wrap your own Python functions

This serializes a Python object or class:

```python
from superduper import model

@model
def my_model(x, y):
    return x + y
```

Additional arguments may be provided to the decorator from `superduper.components.model.ObjectModel`:

```python
@model(num_workers=4)
def my_model(x, y):
    return x + y
```

Similarly the following snippet saves the source code of a python object instead of serializing the object:

```python
from superduper import code

@code
def my_other_model(x, y):
    return x * y
```

These decorators may also be applied to `callable` classes.
If your class has important state which should be serialized with the class, 
then use `model` otherwise you can use `codemodel`:

```python
from superduper import ObjectModel, model

@model
class MyClass:
    def __call__(self, x):
        return x + 2

m = MyClass()

assert isinstance(m, ObjectModel)
assert m.predict(2) == 4
```

As before, additional arguments can be supplied to the decorator:

```python
from superduper import vector, DataType, model

@model(datatype=vector(shape=(32, )))
class MyClass:
    def __call__(self, x):
        return [x + 2] * 32

m = MyClass()

assert isinstance(m.datatype, DataType)
```

## Import classes and functions directly

This may be more intuitive to some developers, 
and allows functionality to be invoked "directly" 
from third-party packages:

```python
from superduper.ext.auto.sklearn.svm import SVC
```

## Create your own `Model` subclasses

Developers may create their own `Model` sub-classes, and deploy these directly to `superduper`.
The key methods the developers need to create are:

- `predict`
- Optionally `predict_batches`, to speed up batching
- Optionally `fit`

### Minimal example with `prediction`

Here is a simple sub-class of `Model`:

```python
from superduper.components.model import Model
import typing as t

class CustomModel(Model):
    signature: t.ClassVar[str] = '**kwargs'
    my_argument: int = 1

    def predict(self, x, y):
        return x + y + self.my_argument
```

The addition of `signature = **kwargs` controls how the individual datapoints in the dataset 
are emitted, for consumption by the internal workings of the model

### Including datablobs which can't be converted to JSON

If your model contains large data-artifacts or non-JSON-able content, then 
these items should be labelled with [a `DataType`](../apply_api/datatype).

On saving, this will allow Superduper to encode their values and save the result
in `db.artifact_store`.

Here is an example which includes a `numpy.array`:

```python
import numpy as np
from superduper.ext.numpy import array


class AnotherModel(Model):
    _artifacts: t.ClassVar[t.Any] = [
        ('my_array', array)
    ]
    signature: t.ClassVar[str] = '**kwargs'
    my_argument: int = 1
    my_array: np.ndarray

    def predict(self, x, y):
        return x + y + self.my_argument + self.my_array

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
and a reference to these `bytes` will be saved in the `db.metadata`.
In principle any `DataType` can be used to encode such an object.
