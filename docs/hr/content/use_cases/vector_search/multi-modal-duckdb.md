---
sidebar_position: 5
---

# Multimodel Vector-Search on DuckDB


SuperDuperDB offers the flexibility to connect to various SQL databases. Check out range of supported SQL databases [here](../../docs/data_integrations/)

In this example, we showcase how to implement multimodal vector-search with DuckDB. This is an extension of multimodal vector-search with MongoDB, which is just slightly easier to set up (see [here](https://docs.superduperdb.com/docs/use_cases/items/multimodal_image_search_clip)). Everything demonstrated here applies equally to any of the supported SQL databases mentioned above, as well as to tabular data formats on disk, such as `pandas`.

Real life use cases could be vectorizing diverse things like images, texts and searching it efficiently with SuperDuperDB.

## Prerequisites

Before proceeding with this use-case, ensure that you have installed the necessary software requirements:

```bash
!pip install superduperdb
```

## Connect to Datastore

The initial step in any `superduperdb` workflow is to connect to your datastore. To connect to a different datastore, simply add a different `URI`, for example, `postgres://...`.

```python
import os
from superduperdb import superduper

os.makedirs('.superduperdb', exist_ok=True)

# Let's super duper your SQL database
db = superduper('duckdb://.superduperdb/test.ddb')
```

## Load Dataset

Now that you're connected, add some data to the datastore:

```python
# Download the coco_sample.zip file
!curl -O https://superduperdb-public.s3.eu-west-1.amazonaws.com/coco_sample.zip

# Download the captions_tiny.json file
!curl -O https://superduperdb-public.s3.eu-west-1.amazonaws.com/captions_tiny.json

# Unzip the contents of coco_sample.zip
!unzip coco_sample.zip

# Create a directory named 'data/coco'
!mkdir -p data/coco

# Move the 'images_small' directory to 'data/coco/images'
!mv images_small data/coco/images
```

```python
# Import necessary libraries
import json
import pandas as pd
from PIL import Image

# Open the 'captions_tiny.json' file and load its contents
with open('captions_tiny.json') as f:
    data = json.load(f)[:500]

# Create a DataFrame from a list comprehension with image paths and captions
data = pd.DataFrame([
    {
        'image': r['image']['_content']['path'],
        'captions': r['captions']
    } for r in data
])

# Add an 'id' column to the DataFrame
data['id'] = pd.Series(data.index).apply(str)

# Create a DataFrame with 'id' and 'image' columns
images_df = data[['id', 'image']]

# Open each image using PIL.Image
images_df['image'] = images_df['image'].apply(Image.open)

# Create a DataFrame with 'id' and 'captions' columns, exploding the 'captions' column
captions_df = data[['id', 'captions']].explode('captions')
```

## Define Schema

For this use-case, you need a table with images and another table with text. SuperDuperDB extends standard SQL functionality, allowing developers to define their own data types through the `Encoder` abstraction.

```python
from superduperdb.backends.ibis.query import Table
from superduperdb.backends.ibis.field_types import dtype
from superduperdb.ext.pillow import pil_image
from superduperdb import Schema

# Define the 'captions' table
captions = Table(
    'captions',
    primary_id='id',
    schema=Schema(
        'captions-schema',
        fields={'id': dtype(str), 'captions': dtype(str)},
    )
)

# Define the 'images' table
images = Table(
    'images',
    primary_id='id',
    schema=Schema(
        'images-schema',
        fields={'id': dtype(str), 'image': pil_image},
    )
)

# Add the 'captions' and 'images' tables to the SuperDuperDB database
db.add(captions)
db.add(images)
```

## Add data to the datastore

```python
# Insert data from the 'images_df' DataFrame into the 'images' table
_ = db.execute(images.insert(images_df))

# Insert data from the 'captions_df' DataFrame into the 'captions' table
_ = db.execute(captions.insert(captions_df))
```

## Build SuperDuperDB `Model` Instances

This use-case utilizes the `superduperdb.ext.torch` extension. Both models used output `torch` tensors, which are encoded with `tensor`:

```python
import clip
import torch
from superduperdb.ext.torch import TorchModel, tensor

# Load the CLIP model
model, preprocess = clip.load("RN50", device='cpu')

# Define a tensor type
t = tensor(torch.float, shape=(1024,))

# Create a TorchModel for text encoding
text_model = TorchModel(
    identifier='clip_text',
    object=model,
    preprocess=lambda x: clip.tokenize(x)[0],
    encoder=t,
    forward_method='encode_text',    
)

# Create a TorchModel for visual encoding
visual_model = TorchModel(
    identifier='clip_image',
    object=model.visual,    
    preprocess=preprocess,
    encoder=t,
)
```

## Create a Vector-Search Index

Define a multimodal search index based on the imported models. The `visual_model` is applied to the images, making the `images` table searchable.

```python
from superduperdb import VectorIndex, Listener

# Add a VectorIndex
db.add(
    VectorIndex(
        'my-index',
        indexing_listener=Listener(
            model=visual_model,
            key='image',
            select=images,
        ),
        compatible_listener=Listener(
            model=text_model,
            key='captions',
            active=False,
            select=None,
        )
    )
)
```

## Search Images Using Text

Now, let's demonstrate how to search for images using text queries:

```python
from superduperdb import Document

# Execute a query to find images with captions containing 'dog catches frisbee'
res = db.execute(
    images
        .like(Document({'captions': 'dog catches frisbee'}), vector_index='my-index', n=10)
        .limit(10)
)
```

Display images

```python
# Display the image data from the fourth result in the search
res[3]['image'].x
```
