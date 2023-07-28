(vectorsearch)=
# Vector Indexes 

```{note}
SuperDuperDB provides first-class support for Vector-Search, including 
encoding of inputs by arbitrary AI models.
```

SuperDuperDB has support for vector-search via LanceDB using vector-indexes.
We are working on support for vector-search via MongoDB enterprise search in parallel.

Vector-indexes build on top of the [datalayer](datalayer), [models](models) and [watchers](watchers).

In order to build a vector index, one defines one or two models, and daemonizes them with watchers.
In the simples variant one does simply:

```python
from superduperdb.core.vector_index import VectorIndex
from sueprduperdb.core.watcher import Watcher

db.add(
    VectorIndex(indexing_watcher='my-model/my-key')
)
```

The model `my-model` should have already been registered with SuperDuperDB (see [models](models) for help). `my-key` is the field to be searched. Together `my-model/my-key` refer to the [watcher](watchers) component (previously created) which is responsible for computing vectors from the data.
See [here](watcher) for how to create such a component.

Alternatively the model and watcher may be created inline. 
Here is how to define a simple libear bag-of-words model:

```python
from superduperdb.core.vector_index import VectorIndex
from superduperdb.core.watcher import Watcher
from superduperdb.models.sentence_transformers.wrapper import Pipeline

class TextEmbedding:
    def __init__(self, lookup):
        self.lookup = lookup   # mapping from strings to pytorch tensors

    def __call__(self, x):
        return sum([self.lookup[y] for y in x.split()])


db.add(
    VectorIndex(
        indexing_watcher=Watcher(
            model=TorchModel(
                preprocess=TextEmbedding(d),     # "d" should be loaded from disk
                object=torch.nn.Linear(64, 512),
            )
        )
    )
)
```
