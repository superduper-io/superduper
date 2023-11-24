# Image Feature Extraction

## Creating a DB of image features in `torchvision`


```python
!pip install superduperdb==0.0.12
!pip install torchvision
```

In this use-case, we demonstrate how to use a pre-trained network from `torchvision` to generate
image features for images which are automatically downloaded into MongoDB. We use a sample 
of the CoCo dataset (https://cocodataset.org/#home) to demonstrate the functionality.


```python
#curl -O https://superduperdb-public.s3.eu-west-1.amazonaws.com/valsmall2014.zip
!unzip -qq valsmall2014.zip
```

As usual, we instantiate the `Datalayer` like this


```python
import os
from superduperdb import superduper
from superduperdb.backends.mongodb import Collection

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

collection = Collection('coco')
```

We then add all of the image URIs to MongoDB. The URIs can be a mixture of local file paths (`file://...`), web URLs (`http...`) and
s3 URIs (`s3://...`). After adding the URIs, SuperDuperDB loads their content into MongoDB - no additional
overhead or job definition required.


```python
import glob
import random

from superduperdb import Document as D
from superduperdb.ext.pillow import pil_image as i

uris = [f'file://{x}' for x in glob.glob('valsmall2014/*.jpg')]

db.execute(collection.insert_many([D({'img': i(uri=uri)}) for uri in uris], encoders=(i,)))
```

We can verify that the images were correctly stored in the `Datalayer`:


```python
from IPython.display import display

# Jupyter often crashes with bigger images
display_image = lambda x: display(x.resize((round(x.size[0] * 0.5), round(x.size[1] * 0.5))))

x = db.execute(collection.find_one())['img'].x

display_image(x)
```

Now let's create the `torch`+`torchvision` model using the `TorchModel` wrapper from SuperDuperDB.
It's possible to create arbitrary pre- and post-processing along with the model forward pass:


```python
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models

import warnings

from superduperdb.ext.torch import TorchModel, tensor

t = transforms.Compose([
    transforms.Resize((224, 224)),   #must same as here
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess(x):
    try:
        return t(x)
    except Exception as e:
        warnings.warn(str(e))
        return torch.zeros(3, 224, 224)

resnet50 = models.resnet50(pretrained=True)
modules = list(resnet50.children())[:-1]
resnet50 = nn.Sequential(*modules)

model = TorchModel(
    identifier='resnet50',
    preprocess=preprocess,
    object=resnet50,
    postprocess=lambda x: x[:, 0, 0],
    encoder=tensor(torch.float, shape=(2048,))
)
```

Let's verify `model` by testing on a single data-point `one=True`:


```python
model.predict(x, one=True)
```

Now that we've got the model ready, we can apply it to the images in the `Datalayer`:


```python
model.predict(
    X='img',
    db=db,
    select=collection.find(),
    batch_size=10,
    max_chunk_size=3000,
    in_memory=False,
    listen=True,
)
```

Let's verify that the features were stored in the `Datalayer`. You can see them in the
`_outputs.img.resnet50` field: 


```python
db.execute(collection.find_one()).unpack()
```
