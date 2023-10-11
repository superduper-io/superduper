# Multimodal search with CLIP

In this notebook we show-case SuperDuperDB's functionality for searching with multiple types of data over
the same `VectorIndex`. This comes out very naturally, due to the fact that SuperDuperDB allows
users and developers to add arbitrary models to SuperDuperDB, and (assuming they output vectors) use
these models at search/ inference time, to vectorize diverse queries.

To this end, we'll be using the [CLIP multimodal architecture](https://openai.com/research/clip).


```python
!pip install git+https://github.com/openai/CLIP
!pip install datasets
!pip install superduperdb
```

So let's start. 

SuperDuperDB supports MongoDB as a databackend. Correspondingly, we'll import the python MongoDB client `pymongo`
and "wrap" our database to convert it to a SuperDuper `Datalayer`:


```python
import os
from superduperdb import CFG
from superduperdb.db.base.build import build_datalayer
from superduperdb.db.mongodb.query import Collection

# Uncomment one of the following lines to use a bespoke MongoDB deployment
# For testing the default connection is to mongomock

mongodb_uri = os.getenv("MONGODB_URI", "mongomock://test")
# mongodb_uri = "mongodb://localhost:27017/documents"
# mongodb_uri = "mongodb+srv://<username>:<password>@<atlas_cluster>/<database>"

# Super-Duper your Database!
CFG.data_backend = mongodb_uri
CFG.artifact_store = 'filesystem://./models'
CFG.vector_search = mongodb_uri

db = build_datalayer(CFG)

collection = Collection(name='tiny-imagenet')
```

In order to make this notebook easy to execute an play with, we'll use a sub-sample of the [Tiny-Imagenet
dataset](https://paperswithcode.com/dataset/tiny-imagenet). 

Everything we are doing here generalizes to much larger datasets, with higher resolution images, without
further ado. For such use-cases, however, it's advisable to use a machine with a GPU, otherwise they'll 
be some significant thumb twiddling to do.

To get the images into the database, we use the `Encoder`-`Document` framework. This allows
us to save Python class instances as blobs in the `Datalayer`, but retrieve them as Python objects.
This makes it far easier to integrate Python AI-models with the datalayer.

To this end, SuperDuperDB contains pre-configured support for `PIL.Image` instances. It's also 
possible to create your own encoders.


```python
from superduperdb.container.document import Document as D
from superduperdb.ext.pillow.image import pil_image as i
from datasets import load_dataset
import random

dataset = load_dataset("zh-plus/tiny-imagenet")['valid']
dataset = [D({'image': i(r['image'])}) for r in dataset]
dataset = random.sample(dataset, 1000)
```

The wrapped python dictionaries may be inserted directly to the `Datalayer`:


```python
db.execute(collection.insert_many(dataset, encoders=(i,)))
```

We can verify that the images are correctly stored:


```python
x = db.execute(collection.find_one()).unpack()['image']
display(x.resize((300, 300 * int(x.size[1] / x.size[0]))))
```

We now can wrap the CLIP model, to ready it for multimodel search. It involves 2 components:

- text-encoding
- visual-encoding

Once we have installed both parts, we will be able to search with both images and text for 
matching items:


```python
import clip
from superduperdb.ext.vector.encoder import vector
from superduperdb.ext.torch.model import TorchModel
import torch

model, preprocess = clip.load("RN50", device='cpu')

e = vector(shape=(1024,))

text_model = TorchModel(
    identifier='clip_text',
    object=model,
    preprocess=lambda x: clip.tokenize(x)[0],
    forward_method='encode_text',
    postprocess=lambda x: x.tolist(),
    encoder=e,
)
```


```python
text_model.predict('This is a test', one=True)
```

Similar procedure with the visual part, which takes `PIL.Image` instances as inputs.


```python
visual_model = TorchModel(
    identifier='clip_image',
    preprocess=preprocess,
    object=model.visual,
    postprocess=lambda x: x.tolist(),
    encoder=e,
)
```


```python
visual_model.predict(x, one=True)
```

Now let's create the index for searching by vector. We register both models with the index simultaneously,
but specifying that it's the `visual_model` which will be responsible for creating the vectors in the database
(`indexing_listener`). The `compatible_listener` specifies how one can use an alternative model to search 
the vectors. By using models which expect different types of index, we can implement multimodal search
without further ado.


```python
from superduperdb.container.vector_index import VectorIndex
from superduperdb.container.listener import Listener

db.add(
    VectorIndex(
        'my-index',
        indexing_listener=Listener(
            model=visual_model,
            key='image',
            select=collection.find(),
        ),
        compatible_listener=Listener(
            model=text_model,
            key='text',
            active=False,
            select=None,
        )
    )
)
```

We can now demonstrate searching by text for images:


```python
import clip
from IPython.display import display
from superduperdb.container.document import Document as D

out = db.execute(
    collection.like(D({'text': 'mushroom'}), vector_index='my-index', n=3).find({})
)
for r in out:
    x = r['image'].x
    display(x.resize((300, 300 * int(x.size[1] / x.size[0]))))
```
