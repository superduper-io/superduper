```python
# installs
!pip install superduperdb
!pip install opencv-python
!pip install git+https://github.com/openai/CLIP.git
```


```python
import cv2
import requests
import os
import numpy as np
from tqdm import tqdm

import pymongo

import superduperdb
from superduperdb import superduper
import glob
from PIL import Image
from superduperdb.ext.pillow.image import pil_image as i

from superduperdb.container.document import Document as D
from superduperdb.container.model import Model
from superduperdb.container.schema import Schema
from superduperdb.db.mongodb.query import Collection
from superduperdb.ext.torch.tensor import tensor
from superduperdb.ext.torch.model import TorchModel
import torch
```


```python
s = Schema(identifier='myschema', fields={'image':i})
```

# Create a superduper`db` instance


```python
import os

# Uncomment one of the following lines to use a bespoke MongoDB deployment
# For testing the default connection is to mongomock

mongodb_uri = os.getenv("MONGODB_URI","mongomock://test")
# mongodb_uri = "mongodb://localhost:27017"
# mongodb_uri = "mongodb://superduper:superduper@mongodb:27017/documents"
# mongodb_uri = "mongodb://<user>:<pass>@<mongo_cluster>/<database>"
# mongodb_uri = "mongodb+srv://<username>:<password>@<atlas_cluster>/<database>"

# Super-Duper your Database!
from superduperdb import superduper
db = superduper(mongodb_uri)
```

# Sample video url json


```python
url = 'https://github.com/SuperDuperDB/superduperdb/assets/138251983/99f35f54-d4b0-40e6-a22d-41043d7bd384'
```


```python
db.execute(Collection('video_frames').insert_many([D({'uri':url})]))
```


```python
class Video2Images:
    def download_video(self, url, output_file):
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download video from {url}")
            return False
        return True

    def save_frames_from_video(self, video_file, output_folder, threshold=10, min_frame_interval=300):
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return
    
        prev_frame = None
        frame_count = 0
    
        os.makedirs(output_folder, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
    
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_timestamp = frame_count // fps
            all_zeros = not np.any(np.asarray(gray_frame))
            frame_count += 1
            if all_zeros:
                continue
            
            if prev_frame is not None:
                
                frame_diff = cv2.absdiff(gray_frame, prev_frame)
                mean_diff = np.mean(frame_diff)
                if mean_diff > threshold and frame_count > min_frame_interval:
                    frame_filename = f"{output_folder}/frame_{current_timestamp}.jpg"
                    cv2.imwrite(frame_filename, frame)
    
            
            prev_frame = gray_frame
    
        cap.release()
        cv2.destroyAllWindows()

    def __call__(self, url, name='landscape.mp4'):
        path = '.videos'
        video_path = os.path.join(path, name)
        self.download_video(url, video_path)
        frame_path = os.path.join(path, 'frames')
        self.save_frames_from_video(video_path, frame_path)
        docs = []
        for image in  glob.glob(f'{frame_path}/*.jpg')[:2]:
            current_timestamp = os.path.split(image)[-1].split('.')[0].split('_')[-1]
            image_array = cv2.imread(image)
            
            doc = {'image': Image.fromarray(image_array[:,:,::-1]), 'current_timestamp': current_timestamp, 'frame_uri':image}
            docs.append(doc)
        return docs
```

# Create a Listener which will continously download video urls and save best frames into other collection.


```python
from superduperdb.container.listener import Listener

!mkdir .videos

video2images = Model(identifier='video2images', object=Video2Images(), flatten=True, model_update_kwargs={'document_embedded':False}, output_schema=s)
db.add(
   Listener(
       model=video2images,
       select=Collection(name='video_frames').find(),
       key='uri',
   )
)
```

# Create CLIP model


```python
import clip
```


```python
model, preprocess = clip.load("RN50", device='cpu')
t = tensor(torch.float, shape=(512,))
```


```python
visual_model = TorchModel(
    identifier='clip_image',
    preprocess=preprocess,
    object=model.visual,
    encoder=t,
)
text_model = TorchModel(
    identifier='clip_text',
    object=model,
    preprocess=lambda x: clip.tokenize(x)[0],
    forward_method='encode_text',
    encoder=t,
    device='cpu',
    preferred_devices=None
)
```

## Create VectorIndex with an indexing and compatible listener


```python
from superduperdb.container.vector_index import VectorIndex
from superduperdb.container.listener import Listener
from superduperdb.ext.openai.model import OpenAIEmbedding
from superduperdb.db.mongodb.query import Collection

db.add(
    VectorIndex(
        identifier='VideoSearchIndex',
        indexing_listener=Listener(
            model=visual_model,
            key='_outputs.uri.video2images.image',
            select=Collection(name='_outputs.uri.video2images').find(),
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

# Test vector search by quering a text against saved frames.


```python
#  We will get the timestamp of the resultant frame and start the video from this timestamp.
out = db.execute(
    Collection('_outputs.uri.video2images').like(D({'text': 'A desert like place.'}), vector_index='VideoSearchIndex', n=1).find()
)
result = [c for c in out]
result = result[0].outputs('uri', 'video2images')
search_timestamp = result['current_timestamp']
```


```python
display(result['image'])
```

# Start the video from the resultant timestamp


```python
from IPython.display import display, HTML
video_html = f"""
<video width="640" height="480" controls>
  <source src="./videos/landscape.mp4" type="video/mp4">
</video>
<script>
var video = document.querySelector('video');
video.currentTime = {search_timestamp};
video.play();
</script>
"""

display(HTML(video_html))
```
