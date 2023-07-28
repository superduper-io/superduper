# Models

```{note}
The SuperDuperDB `Model` wrapper is an extension of standard Python-AI-ecosystem models.
`Model` adds preprocessing, postprocessing, and communication with the `Datalayer`
to the standard toolkit.
```

In SuperDuperDB, the primary mode to integrating a new AI model, framework or API provider
is via the `Model` abstraction. This maybe thought of in the following way:

- A wrapper around AI frameworks facilitating use of models with the datalayer.
- A self contained unit of functionality handling all communication with the datalayer.
- A unifying abstraction bringing all AI frameworks onto a single playing field.
- An trainable, parametrizable extension of UDFs from traditional databasing.

The uniformity of the abstraction is inspired by the Sklearn `.fit`, `.predict` paradigm,
but with additional features which are required in order to allow the models to read and
write to the datalayer. We extend this paradigm to frameworks, which have taken their 
own path in API design, bringing all frameworks into a single world of terminology and
functionality.

## Supported frameworks

The following frameworks are supported natively by SuperDuperDB:

- `sklearn`
- `torch`
- `transformers`
- `openai`

The key class is located in `superduperdb.<framework>.<Framework>Model`:

- `superduperdb.sklearn.SklearnModel.`
- `superduperdb.torch.TorchModel.`
- `superduperdb.transformers.TransformersModel.`
- `superduperdb.openai.OpenAIModel.`

## Porting models to SuperDuperDB

`superduper(model)` provides a shortcut to importing the model class directly:

```python
from superduperdb import superduper
from sklearn.svm import SVM
from torch.nn import Linear
from transformers import pipeline

@superduper
def my_custom_function(x):
    return (x + 2) ** 2

svm = superduper(SVM())
linear = superduper(Linear(10, 20))
pipeline = superduper(pipeline...)
```

## Applying models to data with `.predict`

All of the models which we created in the previous step are now ready to be applied to the database:

```python
>>> from superduperdb.datalayer.mongodb.query import Collection
>>> coll = Collection('my_data')
>>> svm.predict(X='x', db=db, select=coll.find())
# Wait a bit
>>> db.execute(coll.find_one())
Document({
    "_id": ObjectId('64b6ba93f8af205501ca7748'),
    'x': Encodable(x=torch.tensor([...])),
    '_outputs': {'x': {'svm': 1}}
})
```

A similar result may be obtained by replaced the `svm` by any of the other models above.

## Training models on data with `.fit`

Using the same analogy to `sklearn` above, SuperDuperDB supports "in-datalayer" training of models:

```python
>>> svm.fit(
    X='x', y='y', db=db, select=coll.find({'_fold': 'train'})
)
# Lots of output corresponding to training here
```

(watchers)=
## Daemonizing models with watchers

Models can be configured so that, when new data is inserted through the SuperDuperDB datalayer, 
then the models spring into action, processing this new data, and repopulating outputs back 
into the datalayer.

```python
>>> model.predict(X='input_col', db=db, select=coll.find(), watch=True)
```

An equivalent syntax is the following:

```python
>>> from superduperdb.core.watcher import Watcher
>>> db.add(
    Watcher(
        model=model,
        select=coll.find(),
        key='input_col',
    )
)
```

After setting up a `Watcher`, whenever data is inserted or updated, jobs are created 
which save the outputs of the model in the `"_outputs"` field.

A `Watcher` may also be configured in [distributed mode](), to watch for changes coming in 
from any sources - i.e. changes are not just detected through the SuperDuperDB datalayer. 
Read more about that [here]().