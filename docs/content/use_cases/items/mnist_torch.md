# Training and maintaining MNIST predictions

In this notebook we'll be implementing a classic machine learning classification task: MNIST hand written digit
recognition, using a convolution neural network, but with a twist: we'll be implementing the task *in database* using SuperDuperDB.


```python
!pip install matplotlib
!pip install superduperdb[torch]
```

SuperDuperDB supports MongoDB as a databackend. Correspondingly, we'll import the python MongoDB client `pymongo`
and "wrap" our database to convert it to a SuperDuper `Datalayer`:


```python
import torch
import torchvision

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

Now that we've connected to SuperDuperDB, let's add some data. MNIST is a good show case for one of the 
key benefits of SuperDuperDB - adding "difficult" data types. This can be done using an `Encoder` 
which is a key wrapper in SuperDuperDB's arsenal. The `Encoder` works closely together with the `Document` 
wrapper. Together they allow Python dictionaries containing non-JSONable/ `bytes` objects, to be insert into
SuperDuperDB:


```python
from superduperdb.ext.pillow.image import pil_image as i
from superduperdb.container.document import Document as D
from superduperdb.db.mongodb.query import Collection

import random

collection = Collection(name='mnist')

mnist_data = list(torchvision.datasets.MNIST(root='./data', download=True))
data = [D({'img': i(x[0]), 'class': x[1]}) for x in mnist_data]
random.shuffle(data)
data = data[:1000]

db.execute(
    collection.insert_many(data[:-100], encoders=[i])
)
```

Now that we've inserted the images and their classes to the database, let's query some data:


```python
r = db.execute(collection.find_one())
r.unpack()
```

When we query the data, it's in exactly the format we inserted it. In particular, we can use the `PIL.Image` instances
to inspect the data:


```python
r.unpack()['img']
```

Now let's create our model. SuperDuperDB supports these frameworks, out-of-the-box:

- `torch`
- `sklearn`
- `transformers`
- `sentence_transformers`
- `openai`
- `langchain`

In this case, we're going to use PyTorch, since it's great for computer vision use-cases.
We can combine `torch` with `torchvision` in SuperDuperDB.


```python
class LeNet5(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = torch.nn.Linear(400, 120)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(120, 84)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

    
def postprocess(x):
    return int(x.topk(1)[1].item())


def preprocess(x):
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
    )(x)
```

We've created `postprocess` and `preprocess` functions to handle the communication with the SuperDuperDB
`Datalayer`. In order to create a native SuperDuperDB model, we wrap the model, preprocessing and postprocessing:


```python
model = superduper(LeNet5(10), preprocess=preprocess, postprocess=postprocess)
db.add(model)
```

The model predicts human readable outputs, directly from the `PIL.Image` objects. All 
models in SuperDuperDB are equipped with a `sklearn`-style `.predict` method. This makes 
it easy to know how each AI-framework will operate in combination with the `Datalayer`.


```python
model.predict([r['img'] for r in data[:10]])
```

Now we're ready to "train" or "fit" the model. Trainable models in SuperDuperDB are equipped 
with a `sklearn`-like `.fit` method:


```python
from torch.nn.functional import cross_entropy

from superduperdb.container.metric import Metric
from superduperdb.container.dataset import Dataset
from superduperdb.ext.torch.model import TorchTrainerConfiguration


job = model.fit(
    X='img',
    y='class',
    db=db,
    select=collection.find(),
    configuration=TorchTrainerConfiguration(
        identifier='my_configuration',
        objective=cross_entropy,
        loader_kwargs={'batch_size': 10},
        max_iterations=10,
        validation_interval=5,
    ),
    metrics=[Metric(identifier='acc', object=lambda x, y: sum([xx == yy for xx, yy in zip(x, y)]) / len(x))],
    validation_sets=[
        Dataset(
            identifier='my_valid',
            select=Collection(name='mnist').find({'_fold': 'valid'}),
        )
    ],
    distributed=False
)
```


```python
from matplotlib import pyplot as plt

model = db.load('model', model.identifier)

plt.plot(model.metric_values['my_valid/acc'])
plt.show()
```

Now that the model has been trained, we can use it to "listen" the data for incoming changes. 
This is set up with a simple predict "on" the database (without loading all the data client-side).

The `listen` toggle "activates" the model:


```python
model.predict(X='img', db=db, select=collection.find(), listen=True, max_chunk_size=100)
```

We can see that predictions are available in `_outputs.img.lenet5`.


```python
db.execute(collection.find_one({'_fold': 'valid'})).unpack()
```

The models "activated" can be seen here:


```python
db.show('listener')
```

We can verify that the model is activated, by inserting the rest of the data:


```python
for r in data[-100:]:
    r['update'] = True

db.execute(collection.insert_many(data[-100:]))
```

You can see that the inserted data, are now also populated with predictions:


```python
db.execute(collection.find_one({'update': True}))['_outputs']
```
