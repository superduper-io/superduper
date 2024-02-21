---
sidebar_position: 3
---


# Video Vector-Search with Text on MongoDB

This notebook guides you through the process of searching for specific textual information within videos and retrieving relevant video segments. To achieve this, we leverage various libraries and techniques, including:
* clip: A library for vision and language understanding.
* PIL: Python Imaging Library for image processing.
* torch: The PyTorch library for deep learning.

Searching within videos with text has practical applications in various domains:

1. **Video Indexing:** People can find specific topics within videos, enhancing search experiences.

2. **Content Moderation:** Social media platforms use text-based searches to identify and moderate content violations.

3. **Content Discovery:** Users search for specific scenes or moments within movies or TV shows using text queries. Security personnel can search within video footage for specific incidents or individuals.

Your imagination is your limit. Basically, all this example is doing is making the video like a blog post and searchable as well!

## Prerequisites

Before diving into the implementation, ensure that you have the necessary libraries installed by running the following commands:

```bash
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
from superduperdb import superduper, Collection, CFG
import os

# Set configuration options for downloads
CFG.downloads.hybrid = True
CFG.downloads.root = './'

# Define the MongoDB URI, with a default value if not provided
mongodb_uri = os.getenv("MONGODB_URI", "mongomock://test")

# SuperDuperDB, now handles your MongoDB database
# It just super dupers your database by initializing a SuperDuperDB datalayer instance with a MongoDB backend and filesystem-based artifact store
db = superduper(mongodb_uri, artifact_store='filesystem://./data/')

# Create a collection named 'videos'
video_collection = Collection('videos')
```

## Load Dataset

We'll begin by configuring a video encoder.

```python
from superduperdb import Encoder

# Create an instance of the Encoder with the identifier 'video_on_file' and load_hybrid set to False
vid_enc = Encoder(
    identifier='video_on_file',
    load_hybrid=False,
)

# Add the Encoder instance to the SuperDuperDB instance
db.add(vid_enc)
```

Let's fetch a sample video from the internet and insert it into our collection.

```python
from superduperdb.base.document import Document

# Insert a video document into the 'videos' collection
db.execute(
    video_collection.insert_one(
        Document({'video': vid_enc(uri='https://superduperdb-public.s3.eu-west-1.amazonaws.com/animals_excerpt.mp4')}) # Encodes the video
    )
)

# Display the list of videos in the 'videos' collection
list(db.execute(Collection('videos').find()))
```

## Register Encoders

Now, let's set up encoders to process videos and extract frames. These encoders will assist in converting videos into individual frames.

```python
import cv2
import tqdm
from PIL import Image
from superduperdb.ext.pillow import pil_image
from superduperdb import Model, Schema

# Define a function to convert a video file into a list of images
def video2images(video_file):
    # Set the sampling frequency for frames
    sample_freq = 10
    
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_file)
    
    # Initialize variables
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    extracted_frames = []
    progress = tqdm.tqdm()

    # Iterate through video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get the current timestamp based on frame count and FPS
        current_timestamp = frame_count // fps
        
        # Sample frames based on the specified frequency
        if frame_count % sample_freq == 0:
            extracted_frames.append({
                'image': Image.fromarray(frame[:,:,::-1]),  # Convert BGR to RGB
                'current_timestamp': current_timestamp,
            })
        frame_count += 1
        progress.update(1)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Return the list of extracted frames
    return extracted_frames

# Create a SuperDuperDB model for the video2images function
video2images_model = Model(
    identifier='video2images',
    object=video2images,
    flatten=True,
    model_update_kwargs={'document_embedded': False},
    output_schema=Schema(identifier='myschema', fields={'image': pil_image})
)
```

Additionally, we'll configure a listener to continually download video URLs and store the best frames in another collection.

```python
from superduperdb import Listener

# Add a listener to process videos using the video2images model
db.add(
   Listener(
       model=video2images,  # Assuming video2images is your SuperDuperDB model
       select=video_collection.find(),
       key='video',
   )
)

# Get the unpacked outputs of the video2images process for a specific video
outputs = db.execute(Collection('_outputs.video.video2images').find_one()).unpack()

# Display the image output from the processed video
image_output = outputs['_outputs']['video']['video2images']['image']
```

## Create CLIP Model

Now, let's establish a model for CLIP (Contrastive Language-Image Pre-training), serving for both visual and textual analysis.

```python
import clip
from superduperdb import vector
from superduperdb.ext.torch import TorchModel

# Load the CLIP model and define a tensor type
model, preprocess = clip.load("RN50", device='cpu')
t = vector(shape=(1024,))

# Create a TorchModel for visual encoding
visual_model = TorchModel(
    identifier='clip_image',
    preprocess=preprocess,
    object=model.visual,
    encoder=t,
    postprocess=lambda x: x.tolist(),
)

# Create a TorchModel for text encoding
text_model = TorchModel(
    identifier='clip_text',
    object=model,
    preprocess=lambda x: clip.tokenize(x)[0],
    forward_method='encode_text',
    encoder=t,
    device='cpu',  # Specify the device for text encoding
    preferred_devices=None,  # Specify preferred devices for model execution
    postprocess=lambda x: x.tolist(),
)
```

## Create VectorIndex

We'll now establish a VectorIndex to index and search the video frames based on both visual and textual content. This includes creating an indexing listener for visual data and a compatible listener for textual data.

```python
from

 superduperdb import Listener, VectorIndex
from superduperdb.backends.mongodb import Collection

# Add a VectorIndex for video search
db.add(
    VectorIndex(
        identifier='video_search_index',
        indexing_listener=Listener(
            model=visual_model, # Visual model for image processing
            key='_outputs.video.video2images.image', # Visual model for image processing
            select=Collection('_outputs.video.video2images').find(), # Collection containing video image data
        ),
        compatible_listener=Listener(
            model=text_model,  # Text model for processing associated text data
            key='text',
            select=None,
            active=False
        )
    )
)
```

## Query Text Against Saved Frames

Now, let's search for something that happened during the video:

```python
# Define the search parameters
search_term = 'Some ducks'
num_results = 1

# Execute the search and get the next result
r = next(db.execute(
    Collection('_outputs.video.video2images')
    .like(Document({'text': search_term}), vector_index='video_search_index', n=num_results)
    .find()
))

# Extract the timestamp from the search result
search_timestamp = r['_outputs']['video']['video2images']['current_timestamp']

# Retrieve the back reference to the original video using the '_source' field
video = db.execute(Collection('videos').find_one({'_id': r['_source']}))
```

## Start the Video from the Resultant Timestamp

Finally, we can display and play the video starting from the timestamp where the searched text is found.

```python
from IPython.display import display, HTML

# Create HTML code for the video player with a specified source and controls
video_html = f"""
<video width="640" height="480" controls>
    <source src="{video['video'].uri}" type="video/mp4">
</video>
<script>
    // Get the video element
    var video = document.querySelector('video');
    
    // Set the current time of the video to the specified timestamp
    video.currentTime = {search_timestamp};
    
    // Play the video automatically
    video.play();
</script>
"""

# Display the HTML code in the notebook
display(HTML(video_html))
```
