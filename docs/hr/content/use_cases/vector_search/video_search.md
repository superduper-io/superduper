---
sidebar_position: 3
---

# Video

## Search within videos with text

## Introduction
This notebook outlines the process of searching for specific textual information within videos and retrieving relevant video segments. To accomplish this, we utilize various libraries and techniques, such as:
* clip: A library for vision and language understanding.
* PIL: Python Imaging Library for image processing.
* torch: The PyTorch library for deep learning.

## Prerequisites

Before diving into the implementation, ensure that you have the necessary libraries installed by running the following commands:


```python
!pip install superduperdb
!pip install ipython opencv-python pillow openai-clip
```

## Connect to datastore 

First, we need to establish a connection to a MongoDB datastore via SuperDuperDB. You can configure the `MongoDB_URI` based on your specific setup. 
Here are some examples of MongoDB URIs:

* For testing (default connection): `mongomock://test`
* Local MongoDB instance: `mongodb://localhost:27017`
* MongoDB with authentication: `mongodb://superduper:superduper@mongodb:27017/documents`
* MongoDB Atlas: `mongodb+srv://<username>:<password>@<atlas_cluster>/<database>`


```python
from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
from superduperdb import CFG
import os

CFG.downloads.hybrid = True
CFG.downloads.root = './'

mongodb_uri = os.getenv("MONGODB_URI","mongomock://test")
db = superduper(mongodb_uri, artifact_store='filesystem://./data/')

video_collection = Collection('videos')
```

## Load Dataset

We'll begin by configuring a video encoder.


```python
from superduperdb import Encoder

vid_enc = Encoder(
    identifier='video_on_file',
    load_hybrid=False,
)

db.add(vid_enc)
```

Now, let's retrieve a sample video from the internet and insert it into our collection.


```python
from superduperdb.base.document import Document

db.execute(video_collection.insert_one(
        Document({'video': vid_enc(uri='https://superduperdb-public.s3.eu-west-1.amazonaws.com/animals_excerpt.mp4')})
    )
)

# Display the list of videos in the collection
list(db.execute(Collection('videos').find()))
```

## Register Encoders

Next, we'll create encoders for processing videos and extracting frames. This encoder will help us convert videos into individual frames.


```python
import cv2
import tqdm
from PIL import Image
from superduperdb.ext.pillow import pil_image
from superduperdb import Model, Schema


def video2images(video_file):
    sample_freq = 10
    cap = cv2.VideoCapture(video_file)

    frame_count = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    extracted_frames = []
    progress = tqdm.tqdm()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_timestamp = frame_count // fps
        
        if frame_count % sample_freq == 0:
            extracted_frames.append({
                'image': Image.fromarray(frame[:,:,::-1]),
                'current_timestamp': current_timestamp,
            })
        frame_count += 1        
        progress.update(1)
    
    cap.release()
    cv2.destroyAllWindows()
    return extracted_frames


video2images = Model(
    identifier='video2images',
    object=video2images,
    flatten=True,
    model_update_kwargs={'document_embedded': False},
    output_schema=Schema(identifier='myschema', fields={'image': pil_image})
)
```

We'll also set up a listener to continuously download video URLs and save the best frames into another collection.


```python
from superduperdb import Listener

db.add(
   Listener(
       model=video2images,
       select=video_collection.find(),
       key='video',
   )
)

db.execute(Collection('_outputs.video.video2images').find_one()).unpack()['_outputs']['video']['video2images']['image']
```

## Create CLIP model
Now, we'll create a model for the CLIP (Contrastive Language-Image Pre-training) model, which will be used for visual and textual analysis.


```python
import clip
from superduperdb import vector
from superduperdb.ext.torch import TorchModel

model, preprocess = clip.load("RN50", device='cpu')
t = vector(shape=(1024,))

visual_model = TorchModel(
    identifier='clip_image',
    preprocess=preprocess,
    object=model.visual,
    encoder=t,
    postprocess=lambda x: x.tolist(),
)

text_model = TorchModel(
    identifier='clip_text',
    object=model,
    preprocess=lambda x: clip.tokenize(x)[0],
    forward_method='encode_text',
    encoder=t,
    device='cpu',
    preferred_devices=None,
    postprocess=lambda x: x.tolist(),
)
```

## Create VectorIndex

We will set up a VectorIndex to index and search the video frames based on both visual and textual content. This involves creating an indexing listener for visual data and a compatible listener for textual data.


```python
from superduperdb import Listener, VectorIndex
from superduperdb.backends.mongodb import Collection

db.add(
    VectorIndex(
        identifier='video_search_index',
        indexing_listener=Listener(
            model=visual_model,
            key='_outputs.video.video2images.image',
            select=Collection('_outputs.video.video2images').find(),
        ),
        compatible_listener=Listener(
            model=text_model,
            key='text',
            select=None,
            active=False
        )
    )
)
```

## Query a text against saved frames.

Now, let's search for something that happened during the video:


```python
# Define the search parameters
search_term = 'Some ducks'
num_results = 1


r = next(db.execute(
    Collection('_outputs.video.video2images').like(Document({'text': search_term}), vector_index='video_search_index', n=num_results).find()
))

search_timestamp = r['_outputs']['video']['video2images']['current_timestamp']

# Get the back reference to the original video
video = db.execute(Collection('videos').find_one({'_id': r['_source']}))
```

## Start the video from the resultant timestamp:

Finally, we can display and play the video starting from the timestamp where the searched text is found.


```python
from IPython.display import display, HTML

video_html = f"""
<video width="640" height="480" controls>
    <source src="{video['video'].uri}" type="video/mp4">
</video>
<script>
    var video = document.querySelector('video');
    video.currentTime = {search_timestamp};
    video.play();
</script>
"""

display(HTML(video_html))
```
