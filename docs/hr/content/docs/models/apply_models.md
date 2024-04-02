---
sidebar_position: 21
---

# Applying the models 

`Model` and `Predictor` instances may be applied directly to data in the database without first fetching the data client-side.

## Procedural API

Applying a model to data, is straightforward with `Model.predict`.

### Out-of-database prediction

As is standard in `sklearn` and other AI libraries and frameworks, such as `tensorflow.keras`,
all `superduperdb` models, support `.predict`, predicting directly on datapoints.
To use this functionality, supply the datapoints directly to the `Model`:

```python
my_model = ...  # code to instantiate model

my_model.predict(X=[<input_datum> for _ in range(num_data_points)])
```

If only a single prediction is desired, then:

```python
my_model.predict(X=<input_datum>, one=True)
```

### In-database, one-time model prediction

It is possible to apply a model directly to the database with `Model.predict`.
In this context, the parameter `X` refers to the field/column of data which is passed to the model.
`X="_base"` passes all of the data (all columns/ fields).

#### MongoDB

```python
my_model = ...  # code to instantiate model

my_model.predict(
    X='<input-field>',
    db=db,
    select=Collection('<my-collection>').find(),
)
```

#### SQL

```python
table = db.load('my-table', 'table_or_collection')

my_model = ...  # code to instantiate model

my_model.predict(
    X='myfield',
    db=db,
    select=table.filter(table.brand == 'Nike').select(table.myfield),
)
```

### In database, daemonized model predictions with `listen=True`

If is also possible to apply a model to create predictions, and also
refresh these predictions, whenever new data comes in:

```python
my_model.predict(
    X='<input-field>',
    db=db,
    select=query,
    listen=True,
)
```

Under-the-hood, this call creates a `Listener` which is deployed on 
the query passed to the `.predict` call.

Read more about the `Listener` abstraction [here](daemonizing_models_with_listeners.md)

### Activating models for vector-search with `create_vector_index=True`

If a model outputs vectors, it is possible to create a `VectorIndex`
in SuperDuperDB, inline, during applying a model:

```python
my_model.predict(
    X='<input-field>',
    db=db,
    select=query,
    create_vector_index=True,
)
```


## Predictions by framework

### Custom

By default, the `Model` component supports arbitrary callables to be used to 
perform model predictions and transformations:

```python
from superduperdb import Model

def chunk_text(x):
    return x.split('\n\n')

m = Model('my-chunker', object=chunk_text)

m.predict(
    X='<input>',
    select=<query>,   # MongoDB, Ibis or SQL query
    db=db,
)
```

### Sklearn

```python
from superduperdb.ext.sklearn import Estimator
from sklearn.svm import SVC

m = Estimator(SVC())

m.predict(
    X='<input>',
    select=<query>,  # MongoDB, Ibis or SQL query
    db=db,
)
```

### Transformers

```python
from superduperdb.ext.transformers import Pipeline
from superduperdb import superduper

m = Pipeline(task='sentiment-analysis')

m.predict(
    X='<input>',
    db=db,
    select=<query>,  # MongoDB, Ibis or SQL query
    batch_size=100,  # any **kwargs supported by `transformers.Pipeline.__call__`
)
```

### PyTorch

```python
import torch
from superduperdb.ext.torch import Module

model = Module(
    'my-classifier',
    preprocess=lambda x: torch.tensor(x),
    object=torch.nn.Linear(64, 512),
    postprocess=lambda x: x.topk(1)[0].item(),
)

model.predict(
    X='<input>',
    db=db,
    select=<query>,  # MongoDB, Ibis or SQL query
    batch_size=100,  # any **kwargs supported by `torch.utils.data.DataLoader`
    num_workers=4,
)
```

### OpenAI

Embeddings:

```python
from superduperdb.ext.openai import OpenAIEmbedding

m = OpenAIEmbedding(identifier='text-embedding-ada-002')

m.predict(
    X='<input>',
    db=db,
    select=<query>,  # MongoDB, Ibis or SQL query
)
```

## Predicting based on the `identifier` via the `Datalayer`

Instead of calling the model directly, it's also possible to 
predict on single data points using `db.predict`.

I.e. the following are equivalent:

```python
my_model = Model('my-model', model_object) # code to instantiate model

my_model.predict(X=<input_datum>)
```

... and

```python
my_model = Model('my-model', model_object) # code to instantiate model

db.add(my_model)
db.predict('my-model', input=<input_datum>)
```

Using `db.predict`, model predictions may be augmented with data from the database.
I.e. the following are equivalent:

```python
db.predict('my-model', input=<input_data>, context_select=<query>)
```

... and

```python
context = db.execute(query)
my_model.predict(<input_data>, context=context)
```

## Models with special outputs

If a model has outputs which aren't directly compatible with the underlying database, then one adds either 
an `Encoder` or a `Schema` to the `Model` at initialization.

Here's a model which outputs images:

```python
from superduperdb.ext.pillow import pil_image

my_model = Model('my-model', model_object, encoder=pil_image)
```

Here's a model which outputs dictionaries with `"img"` (images) and `"txt"` (string) fields:

```python
from superduperdb import Schema

schema = Schema('my-schema', fields={'img': pil_image, 'txt': 'str'})
my_model = Model('my-model', model_object, schema=schema)
```