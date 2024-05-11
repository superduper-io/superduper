# Listening for new data

In SuperDuperDB, AI models may be configured to listen for newly inserted data.
Outputs will be computed over that data and saved back to the data-backend.

In this example we show how to configure 3 models to interact when new data is added.

1. A featurizing computer vision model (images `->` vectors).
1. 2 models evaluating image-2-text similarity to a set of key-words.


```python
!curl -O https://superduperdb-public-demo.s3.amazonaws.com/images.zip && unzip images.zip
from PIL import Image

data = [f'images/{x}' for x in os.listdir('./images')]
data = [Image.open(path) for path in data]
sample_datapoint = data[-1]
```


```python
from superduperdb import superduper

db = superduper('mongomock://')

db['images'].insert_many(data)
```


```python
import torch
import clip
from torchvision import transforms
from superduperdb import ObjectModel
from superduperdb import Listener

import torch
import clip
from PIL import Image


class CLIPModel:
    def __init__(self):
        # Load the CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("RN50", device=self.device)

    def __call__(self, text, image):
        with torch.no_grad():
            text = clip.tokenize([text]).to(self.device)
            image = self.preprocess(Image.fromarray(image.astype(np.uint8))).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image)[0].numpy().tolist()
            text_features = self.model.encode_text(text)[0].numpy().tolist()
        return [image_features, text_features]
        

model = ObjectModel(
    identifier="clip",
    object=CLIPModel(),
    signature="**kwargs",
)
```


```python
listener = model.to_listener(
    select=db['images'].find(),
    key='image',
    identifier='image_predictions',
)

db.apply(listener)
```

```python
words = ['hat', 'cat', 'mat']

targets = {word: model.predict_one(word) for word in words}

class Comparer:
    def __init__(self, targets):
        self.targets = targets
        self.lookup = list(self.targets.keys())
        self.matrix = torch.stack(list(self.targets.values()))

    def __call__(self, vector):
        best = (self.matrix @ vector).topk(1)[1].item()
        return self.lookup[best]

comparer = ObjectModel(
    'comparer',
    object=Comparer(targets)).to_listener(
        select=db['images'].find(), 
        key=f'_outputs.{listener.uuid}'
    ),
)

db.apply(comparer)
```
