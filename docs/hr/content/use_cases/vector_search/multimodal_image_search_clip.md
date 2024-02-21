---
sidebar_position: 2
---


#  Multimodal Vector-Search Using CLIP on MongoDB

This notebook demonstrates how SuperDuperDB can perform multimodal searches using the `VectorIndex`. It highlights SuperDuperDB's flexibility in integrating different models for vectorizing diverse queries during search and inference. In this example, we utilize the [CLIP multimodal architecture](https://openai.com/research/clip).

Real life use cases could be vectorizing diverse things like images and searching it efficiently.

## Prerequisites

Before starting, make sure you have the required libraries installed. Run the following commands:

```bash
!pip install superduperdb
!pip install ipython openai-clip
!pip install -U datasets
```

## Connect to datastore

First, we need to establish a connection to a MongoDB datastore via SuperDuperDB. You can configure the `MongoDB_URI` based on your specific setup.
Here are some examples of MongoDB URIs:

- For testing (default connection): `mongomock://test`
- Local MongoDB instance: `mongodb://localhost:27017`
- MongoDB with authentication: `mongodb://superduper:superduper@mongodb:27017/documents`
- MongoDB Atlas: `mongodb+srv://<username>:<password>@<atlas_cluster>/<database>`

```python
import os
from superduperdb import superduper
from superduperdb.backends.mongodb import Collection

mongodb_uri = os.getenv("MONGODB_URI", "mongomock://test")
db = superduper(mongodb_uri, artifact_store='filesystem://./models/')

# SuperDuperDB, now handles your MongoDB database
# It just super dupers your database 
db = superduper(mongodb_uri, artifact_store='filesystem://.data')

collection = Collection('multimodal')
```

## Load Dataset

For simplicity and interactivity, we'll use a subset of the [Tiny-Imagenet dataset](https://paperswithcode.com/dataset/tiny-imagenet). The processes shown here can be applied to larger datasets with higher-resolution images. If working with larger datasets, especially with high-resolution images, it's recommended to use a machine with a GPU for efficiency.

To insert images into the database, we'll use the `Encoder`-`Document` framework. This framework allows saving Python class instances as blobs in the `Datalayer` and retrieving them as Python objects. SuperDuperDB comes with built-in support for `PIL.Image` instances, making it easy to integrate Python AI models with the datalayer. If needed, you can also create custom encoders.

```bash
!curl -O https://superduperdb-public.s3.eu-west-1.amazonaws.com/coco_sample.zip
!unzip coco_sample.zip
```

```python
from superduperdb import Document
from superduperdb.ext.pillow import pil_image as i
import glob
import random

# Use glob to get a list of image file paths in the 'images_small' directory
images = glob.glob('images_small/*.jpg')

# Create a list of SuperDuperDB Document instances with image data
# Note: The 'uri' parameter is set to the file URI using the 'file://' scheme
# The list is limited to the first 500 images for demonstration purposes
documents = [Document({'image': i(uri=f'file://{img}')}) for img in images][:500]
```

Access a random Document in the documents list, just to check:

```python
# Access a random Document in the documents list, just to check
documents[1]
```

The wrapped python dictionaries may be inserted directly to the `Datalayer`:

```python
# Insert the list of Document instances into a collection using SuperDuperDB
# Specify the 'i' encoder (pil_image) for the 'image' field
db.execute(collection.insert_many(documents), encoders=(i,))
```

You can verify that the images are correctly stored as follows:

```python
x = db.execute(imagenet_collection.find_one()).unpack()['image']

# Resize the image for display while maintaining the aspect ratio and Display the resized image
display(x.resize((300, 300 * int(x.size[1] / x.size[0]))))
```

## Build Models

Now, let's prepare the CLIP model for multimodal search. This involves two components: `text encoding` and `visual encoding`. Once both components are installed, you can perform searches using both images and text to find matching items.

```python
import clip
from superduperdb import vector
from superduperdb.ext.torch import TorchModel

# Load the CLIP model and obtain the preprocessing function
model, preprocess = clip.load("RN50", device='cpu')

# Define a vector with shape (1024,)
e = vector(shape=(1024,))

# Create a TorchModel for text encoding
text_model = TorchModel(
    identifier='clip_text', # Unique identifier for the model
    object=model, # CLIP model
    preprocess=lambda x: clip.tokenize(x)[0],  # Model input preprocessing using CLIP 
    postprocess=lambda x: x.tolist(), # Convert the model output to a list
    encoder=e,  # Vector encoder with shape (1024,)
    forward_method='encode_text', # Use the 'encode_text' method for forward pass 
)

# Create a TorchModel for visual encoding
visual_model = TorchModel(
    identifier='clip_image',  # Unique identifier for the model
    object=model.visual,  # Visual part of the CLIP model    
    preprocess=preprocess, # Visual preprocessing using CLIP
    postprocess=lambda x: x.tolist(), # Convert the output to a list 
    encoder=e, # Vector encoder with shape (1024,)
)
```

## Create a Vector-Search Index

Now, let's create the index for vector-based searching. We'll register both models with the index simultaneously. Specify that the `visual_model` will be responsible for creating vectors in the database (`indexing_listener`). The `compatible_listener` indicates how an alternative model can be used to search the vectors, allowing multimodal search with models expecting different types of indexes.

```python
from superduperdb import VectorIndex
from superduperdb import Listener

# Create a VectorIndex and add it to the database
db.add(
    VectorIndex(
        'my-index', # Unique identifier for the VectorIndex
        indexing_listener=Listener(
            model=visual_model, # Visual model for embeddings
            key='image', # Key field in documents for embeddings
            select=collection.find(), # Select the documents for indexing
            predict_kwargs={'batch_size': 10}, # Prediction arguments for the indexing model
        ),
        compatible_listener=Listener(
            # Create a listener to listen upcoming changes in databases
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

# Execute the 'like' query using the VectorIndex 'my-index' and find the top 3 results
out = db.execute(
    collection.like(Document

({'text': query_string}), vector_index='my-index', n=3).find({})
)

# Display the images from the search results
for r in search_results:
    x = r['image'].x
    display(x.resize((300, int(300 * x.size[1] / x.size[0]))))
```

Let's dig further:

```python
img = db.execute(collection.find_one({}))['image']
img.x
```

## Now let's try Similarity search

Perform a similarity search using the vector index 'my-index'
Find the top 3 images similar to the input image 'img'
Finally displaying the retrieved images while resizing them for better visualization.

```python
# Execute the 'like' query using the VectorIndex 'my-index' to find similar images to the specified 'img'
similar_images = db.execute(
    collection.like(Document({'image': img}), vector_index='my-index', n=3).find({})
)

# Display the similar images from the search results
for i in similar_images:
    x = i['image'].x
    display(x.resize((300, int(300 * x.size[1] / x.size[0]))))
```