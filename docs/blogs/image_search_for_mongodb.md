
Hi,

sharing with you a draft of the blog post. It is not finished, but sharing it now as I won't have time to work on it any more on it this week.

It is a couple of edit iterations away from being publishable. It is probably too long for a single post, so plan to split it into two or three that are short and punchy.

-----

Target audience:
- e-commerce web developers, who are somewhat familiar with python
- python developers, not familiar with data science
- data scientists, who are not familiar with image search

----
# Easy E-Commerce Image Search for MongoDB

Abstract: How to easily move beyond the limited keyword search through product catalog, towards a semantic search that understand the meaning of images. While at the same time staying in the familiar MongoDB ecosystem.

## Introduction
User Problem: searching for the product in online store needs manual tagging and i have to know the tag, like I search for a fork and find nothing. I had to search for  "cutlery" because that is how it was manually tagged in product catalog. (**TODO:** add image examples like in this blog: https://wasimlorgat.com/posts/image-vector-search.html )

The solution to this user problem is to stop matching search keywords to the manual labels attached to products, and instead focus on understanding the meaning of the image and text. This is called "semantic search". We represent the meaning of a text or an image as a list of numbers ("a vector"). If two vectors are similar, then the meaning is the same. When a user searching for "a fork" we would return all the products that have an image vector similar to the vector for the word "fork".

## What is CLIP?

How to build a good correspondence between vectors and words and pixels?

Contrastive Language-Image Pretraining (CLIP) is a technique for training neural networks to do exactly that.

OpenAI has collected millions of images from the Web together with their alt-text, and trained a neural network to put the image vector close to the vector corresponding to the image caption, and far away from other captions and images. 

![[Pasted image 20230726120114.png]]
(image from https://www.pinecone.io/learn/series/image-search/clip/)

## Technical challenges

Great, so now we can just take the model open-sourced by OpenAI in a [Python package](https://github.com/openai/CLIP) and run it on our product catalog.

Unfortunately there are still technical challenges here: 
- Collecting all the images in one place.
- Run the model to get vectors for all the images already in the catalog
- Automatically create vectors for the new images, the moment they get added to the catalog
- Find products with similar vectors when a text search query comes in
- Serve them to the website front end code

The usual way to do this is to setup a AWS Cloudformation templates to pre-process the images, run CLIP on them. All of this needs to be orchestrated via AirFlow DAG and added to the pipeline for adding new data. 

To find similar vectors you will need to setup a new service - a vector database. There are plenty of them to choose from. 
The backend search API endpoint needs to have a new workflow for serving a request. A text comes in, it is converted to a vector with CLIP, then a query is sent to the vector database to find ids of products similar vectors, and then these ids need to be retrieved from the product catalog and served back.
## SuperDuperDB makes light work

You can overcome these technical challenges in <100 lines with SuperDuperDB, see code [here](./notebooks/blog_version_multimodal_image_search_clip.ipynb).

I will go into detail below, but at high level:
- stores images as binary data in MongoDB;
- get image vectors from CLIP, parallelised via Dask
- store vectors in a vector database LanceDB
- create a watcher thread that watches the database for new images and creates vectors for them automatically

(**TODO:** this diagram is just a placeholder, make a smaller and more concise architecture diagram: Mongo, Lance, inference thread.)![[Screenshot 2023-07-27 at 12.28.36.png]]
### Store images in MongoDB

We start by taking the images from a dataset on disk and storing them in MongoDB. 

By the way, getting to a stage of "dataset on disk" or "data in S3 bucket" might not be so easy, as they can be in many different places and micro-services. SuperDuperDB has easy ways to retrieve and process a remote resource URI. (**TODO**: link to docs or code )

So here is the code snippet.

(**TODO**: maybe make naming convention more aligned in the repo - camelcase or underscores, Document or pil_image )

```python
from superduperdb.core.document import Document as SuperDuperDocument

from superduperdb.encoders.pillow.image import pil_image as image_encoder

sdp_documents = [SuperDuperDocument({'image': image_encoder(r['image'])}) for r in dataset]

db.execute(collection.insert_many(sdp_documents, encoders=(image_encoder,)))
```

We take an image from the dataset, encode it as a binary object for easy storage in Mongo. 

You can inspect the data with a nice [MongoDB Compass](https://www.mongodb.com/try/download/compass) desktop UI. This tool is one of the many advantages of staying within the MongoDB ecosystem.

![[Screenshot 2023-07-27 at 11.55.08.png]]

SuperDuperDB has encoders for easy storage of many tricky datatypes - audio files, PyTorch tensors, vector embeddings.

# Create vectors from the images and text

Now we load a specific version of CLIP that runs on CPU via PyTorch. We use a SuperDuperDB PyTorch tensor encoder to store the model output in the DB. 

```python
import torch
import clip

from superduperdb.encoders.torch.tensor import tensor as tensor_encoder
from superduperdb.models.torch.wrapper import TorchModel

my_tensor_encoder = tensor_encoder(torch.float, shape=(512,))

model, preprocess = clip.load("ViT-B/32", device='cpu')

text_model = TorchModel(
	identifier='clip_text',
	object=model,
	preprocess=lambda x: clip.tokenize(x)[0],
	forward_method='encode_text',
	encoder=my_tensor_encoder
)

visual_model = TorchModel(
	identifier='clip_image',
	preprocess=preprocess,
	object=model.visual,
	encoder=my_tensor_encoder,
)
```

Next is adding the vectors for each image already in the collection. And making sure they are added for the new images automatically via an active indexing watcher.
We also add a possibility to search by 'text' via an in-active `compatible_watcher`.

```python
from superduperdb.core.vector_index import VectorIndex
from superduperdb.core.watcher import Watcher

db.add(
    VectorIndex(
        'my-index',
        indexing_watcher=Watcher(
            model=visual_model,
            key='image',
            select=collection.find(),
        ),
        compatible_watcher=Watcher(
            model=text_model,
            key='text',
            active=False,
        )
    )
)```

Here is how the model output is stored in Mongo. Any model you run is stored in in the `_outputs` field of the document that it took as input. The outputs will be versioned for each model run in the next releases, watch this space.

```json
{
  "_id": {
    "$oid": "64c23c9ad761f1b1ca002444"
  },
  "image": {
    "_content": {
      "bytes": {
        "$binary": {
          "base64": "iVBORw0KGgo...",
          "subType": "00"
        }
      },
      "encoder": "pil_image"
    }
  },
  "_fold": "train",
  "_outputs": {
    "image": {
      "clip_image": {
        "_content": {
          "bytes": {
            "$binary": {
              "base64": "vyc7vEK/nL4oM...",
              "subType": "00"
            }
          },
          "encoder": "torch.float32[512]"
        }
      }
    }
  }
}
```

## Serving search results to the Web Front End

Now we can produce a good list of images that are related by meaning to the text that the user is searching for, not just by some manually-assigned catalog tag.
How do we serve them back to the application?

Here is a Python code snippet from the interactive demo in this [notebook](notebooks/multimodal_image_search_clip.ipynb). 
(**TODO:** fix the formatting of the first line of code in the code snippet example)
![[Screenshot 2023-07-26 at 16.56.24.png]]

It produces a cursor, similar to the usual MongoDB PyMongo cursor. 

You can expose it as REST API using a FastAPI and serve via React front end, as in this [demo](https://github.com/weaviate/weaviate-examples/tree/main/clip-multi-modal-text-image-search) (**TODO:** build analogue to this Weaviate demo but with superduperDB,  show model confidence, add new image upload capability.)
![[Screenshot 2023-07-26 at 11.47.51.png]]


If you are already on the [FARM Web-stack](https://www.mongodb.com/developer/languages/python/farm-stack-fastapi-react-mongodb/) (FastAPI, React, and MongoDB), then this is easy. We are working on a REST API so you can use SuperDuperDB vector index with other web backends, Node.js or no code Web frameworks like Bubble (**TODO:** add it to the roadmap?. ). 

## Next Blog post: Training and Comparing models

In my next blog post I will talk about how to make the search work even better for your particular online store.

For example, you have a great niche store that sells candles. CLIP out of the box can distinguish between a candle and candlestick, but it can't tell apart "tea light" and "pillar" kinds of candle.

It is easy to train CLIP on your data with a few lines of code with SuperDuperDB. Depending on how many images you have, this training takes a couple of days to run on a GPU. Before committing to this time and cost, you need to be sure that your training code is rock solid. 

SuperDuperDB also makes it easy to compare two models on the same data.

Just to illustrate how easy it is to do qualitative comparison of two models with SuperDuperDB. We add two vector-indices to the same MongoDB, search them simultaneously and compare results in a Gradio UI [here](notebooks/compare_multimodal_image_search_clip_openclip.ipynb) . The two models here have identical CLIP architecture. The difference is that one was trained by OpenAI on their closed-source dataset and the other on an open-source [LAION 2B](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K). 

![[Screenshot 2023-07-26 at 15.19.56.png]]

## Conclusion

