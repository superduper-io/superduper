# Stack

- Wraps multiple potentially interdependent components
- Allows developers to "apply" a range of functionality in a single declaration
- Allows decision makers and admins to manage "groups" of AI functionality

***Usage pattern***

```python
from superduperdb import Stack

stack = Stack(
    'my-stack',
    components=[
        table_1,
        model_1,
        model_2,
        listener_1,
        vector_index_1,
    ]
)

db.apply(m)
```


Here is an example in which vectors are prepared using a 
convolutional neural network over images, 
and these vectors are used downstream in ***both***
vector-search and in a transfer-learning task.

1. The `Listener` instance, wraps the CNN `'my-cnn-vectorizer'`,
which contains the `torch` layer and pre-processing/ post-processing.

2. The `Stack` reuses this `Listener` twice, once in the `VectorIndex`,
which may be used to find images, using images,
and once with the support-vector-machine `SVC()`, which ingests 
the vectors calculated by the `Listener`, and, is fitted
based on those vectors and the label set.

```python
from sklearn.svm import SVC
from my_models.vision import MyTorchModule, prepare_image

from superduperdb.ext.numpy import array
from superduperdb.ext.sklearn import Estimator, SklearnTrainer
from superduperdb.ext.torch import TorchModel
from superduperdb import Stack, VectorIndex, Listener


my_listener=Listener(
    'my-listener',
    model=TorchModel(
        'my-cnn-vectorizer',
        object=MyTorchModule(),
        preprocess=prepare_image,
        postprocess=lambda x: x.numpy(),
        encoder=array(dtype='float', shape=(512,)),
    )
    key='img',
    select=db['documents'].find({'_fold': 'train'}),
)

db.apply(
    Stack(
        'my-stack',
        [
            my_listener,
            VectorIndex(
                'my-index',
                indexing_listener=my_listener,
            ),
            Estimator(
                'my-classifier',
                object=SVC()
                postprocess=lambda x: ['apples', 'pears'][x]
                trainer=SklearnTrainer(
                    'my-trainer',
                    select=my_listener.select,
                    key=(my_listener.outputs, labels),
                )
            )
        ],
    )
)
```

***See also***

- [YAML stack syntax](../production/yaml_formalism.md)