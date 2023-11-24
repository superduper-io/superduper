---
sidebar_position: 2
---
# Image

## Multimodal Search Using CLIP

This notebook showcases the capabilities of SuperDuperDB for performing multimodal searches using the `VectorIndex`. SuperDuperDB's flexibility enables users and developers to integrate various models into the system and use them for vectorizing diverse queries during search and inference. In this demonstration, we leverage the [CLIP multimodal architecture](https://openai.com/research/clip).

## Prerequisites

Before diving into the implementation, ensure that you have the necessary libraries installed by running the following commands:


```python
!pip install superduperdb
!pip install ipython openai-clip
!pip install -U datasets
```

## Connect to datastore 

First, we need to establish a connection to a MongoDB datastore via SuperDuperDB. You can configure the `MongoDB_URI` based on your specific setup. 
Here are some examples of MongoDB URIs:

* For testing (default connection): `mongomock://test`
* Local MongoDB instance: `mongodb://localhost:27017`
* MongoDB with authentication: `mongodb://superduper:superduper@mongodb:27017/documents`
* MongoDB Atlas: `mongodb+srv://<username>:<password>@<atlas_cluster>/<database>`


```python
import os
from superduperdb import superduper
from superduperdb.backends.mongodb import Collection

mongodb_uri = os.getenv("MONGODB_URI", "mongomock://test")
db = superduper(mongodb_uri, artifact_store='filesystem://./models/')

# Super-Duper your Database!
db = superduper(mongodb_uri, artifact_store='filesystem://.data')

collection = Collection('multimodal')
```

## Load Dataset 

To make this notebook easily executable and interactive, we'll work with a sub-sample of the [Tiny-Imagenet dataset](https://paperswithcode.com/dataset/tiny-imagenet). The processes demonstrated here can be applied to larger datasets with higher resolution images as well. For such use-cases, however, it's advisable to use a machine with a GPU, otherwise they'll be some significant thumb twiddling to do.

To insert images into the database, we utilize the `Encoder`-`Document` framework, which allows saving Python class instances as blobs in the `Datalayer` and retrieving them as Python objects. To this end, SuperDuperDB contains pre-configured support for `PIL.Image` instances. This simplifies the integration of Python AI models with the datalayer. It's also possible to create your own encoders.



```python
!curl -O https://superduperdb-public.s3.eu-west-1.amazonaws.com/coco_sample.zip
!unzip coco_sample.zip
```


```python
from superduperdb import Document
from superduperdb.ext.pillow import pil_image as i
import glob
import random

images = glob.glob('images_small/*.jpg')
documents = [Document({'image': i(uri=f'file://{img}')}) for img in images][:500]
```


```python
documents[1]
```

The wrapped python dictionaries may be inserted directly to the `Datalayer`:


```python
db.execute(collection.insert_many(documents), encoders=(i,))
```

You can verify that the images are correctly stored as follows:


```python
x = db.execute(imagenet_collection.find_one()).unpack()['image']
display(x.resize((300, 300 * int(x.size[1] / x.size[0]))))
```

## Build Models
We now can wrap the CLIP model, to ready it for multimodal search. It involves 2 components:

Now, let's prepare the CLIP model for multimodal search, which involves two components: `text encoding` and `visual encoding`. After installing both components, you can perform searches using both images and text to find matching items:


```python
import clip
from superduperdb import vector
from superduperdb.ext.torch import TorchModel

# Load the CLIP model
model, preprocess = clip.load("RN50", device='cpu')

# Define a vector
e = vector(shape=(1024,))

# Create a TorchModel for text encoding
text_model = TorchModel(
    identifier='clip_text',
    object=model,
    preprocess=lambda x: clip.tokenize(x)[0],
    postprocess=lambda x: x.tolist(),
    encoder=e,
    forward_method='encode_text',    
)

# Create a TorchModel for visual encoding
visual_model = TorchModel(
    identifier='clip_image',
    object=model.visual,    
    preprocess=preprocess,
    postprocess=lambda x: x.tolist(),
    encoder=e,
)
```

## Create a Vector-Search Index

Let's create the index for vector-based searching. We'll register both models with the index simultaneously, but specify that the `visual_model` will be responsible for creating the vectors in the database (`indexing_listener`). The `compatible_listener` specifies how an alternative model can be used to search the vectors, enabling multimodal search with models expecting different types of indexes.


```python
from superduperdb import VectorIndex
from superduperdb import Listener

# Create a VectorIndex and add it to the database
db.add(
    VectorIndex(
        'my-index',
        indexing_listener=Listener(
            model=visual_model,
            key='image',
            select=collection.find(),
            predict_kwargs={'batch_size': 10},
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

## Search Images Using Text

Now we can demonstrate searching for images using text queries:


```python
from IPython.display import display
from superduperdb import Document

query_string = 'sports'

out = db.execute(
    collection.like(Document({'text': query_string}), vector_index='my-index', n=3).find({})
)

# Display the images from the search results
for r in search_results:
    x = r['image'].x
    display(x.resize((300, int(300 * x.size[1] / x.size[0]))))
```


```python
img = db.execute(collection.find_one({}))['image']
img.x
```


```python
cur = db.execute(
    collection.like(Document({'image': img}), vector_index='my-index', n=3).find({})
)

for r in cur:
    x = r['image'].x
    display(x.resize((300, int(300 * x.size[1] / x.size[0]))))
```


```python

```
