# Training and Maintaining MNIST Predictions with SuperDuperDB

## Introduction

This notebook outlines the process of implementing a classic machine learning classification task - MNIST handwritten digit recognition, using a convolutional neural network. However, we introduce a unique twist by performing the task in a database using SuperDuperDB.

## Prerequisites

Before diving into the implementation, ensure that you have the necessary libraries installed by running the following commands:


```python
!pip install superduperdb
!pip install torch torchvision matplotlib
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
import os

mongodb_uri = os.getenv("MONGODB_URI","mongomock://test")
db = superduper(mongodb_uri)

# Create a collection for MNIST
mnist_collection = Collection('mnist')
```


## Load Dataset

After connecting to MongoDB, we add the MNIST dataset. SuperDuperDB excels at handling "difficult" data types, and we achieve this using an `Encoder`, which works in tandem with the `Document` wrappers. Together, they enable Python dictionaries containing non-JSONable or bytes objects to be inserted into the underlying data infrastructure. 



```python
import torchvision
from superduperdb.ext.pillow import pil_image
from superduperdb import Document
from superduperdb.backends.mongodb import Collection

import random

# Load MNIST images as Python objects using the Python Imaging Library.
mnist_data = list(torchvision.datasets.MNIST(root='./data', download=True))
document_list = [Document({'img': pil_image(x[0]), 'class': x[1]}) for x in mnist_data]

# Shuffle the data and select a subset of 1000 documents
random.shuffle(document_list)
data = document_list[:1000]

# Insert the selected data into the mnist_collection
db.execute(
    mnist_collection.insert_many(data[:-100]),  # Insert all but the last 100 documents
    encoders=(pil_image,) # Encode images using the Pillow library.
)
```

Now that the images and their classes are inserted into the database, we can query the data in its original format. Particularly, we can use the `PIL.Image` instances to inspect the data.


```python
# Get and display one of the images
r = db.execute(mnist_collection.find_one())
r.unpack()['img']
```

## Build Model

Next, we create our machine learning model. SuperDuperDB supports various frameworks out of the box, and in this case, we are using PyTorch, which is well-suited for computer vision tasks. In this example, we combine torch with torchvision.

We create `postprocess` and `preprocess` functions to handle the communication with the SuperDuperDB `Datalayer`, and then wrap model, preprocessing and postprocessing to create a native SuperDuperDB handler.



```python
import torch

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


# Create and insert a SuperDuperDB model into the database
model = superduper(LeNet5(10), preprocess=preprocess, postprocess=postprocess, preferred_devices=('cpu',))
db.add(model)
```

## Train Model

Now we are ready to "train" or "fit" the model. Trainable models in SuperDuperDB come with a sklearn-like `.fit` method. 



```python
from torch.nn.functional import cross_entropy

from superduperdb import Metric
from superduperdb import Dataset
from superduperdb.ext.torch.model import TorchTrainerConfiguration

# Fit the model to the training data
job = model.fit(
    X='img', # Feature matrix used as input data 
    y='class', # Target variable for training
    db=db, # Database used for data retrieval
    select=mnist_collection.find(), # Select the dataset
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
            select=Collection('mnist').find({'_fold': 'valid'}),
        )
    ],
    distributed=False,
)
```

## Monitoring Training Efficiency
You can monitor the training efficiency with visualization tools like Matplotlib:


```python
from matplotlib import pyplot as plt

# Load the model from the database
model = db.load('model', model.identifier)

# Plot the accuracy values
plt.plot(model.metric_values['my_valid/acc'])
plt.show()
```


## On-the-fly Predictions
Once the model is trained, you can use it to continuously predict on new data as it arrives. This is set up by enabling a `listener` for the database (without loading all the data client-side). The listen toggle activates the model to make predictions on incoming data changes.



```python
model.predict(
    X='img', # Input feature  
    db=db,  # Database used for data retrieval
    select=mnist_collection.find(), # Select the dataset
    listen=True, # Continuous predictions on incoming data 
    max_chunk_size=100, # Number of predictions to return at once
)
```

We can see that predictions are available in `_outputs.img.lenet5`.


```python
r = db.execute(mnist_collection.find_one({'_fold': 'valid'}))
r.unpack()
```

## Verification

The models "activated" can be seen here:


```python
db.show('listener')
```

We can verify that the model is activated, by inserting the rest of the data:


```python
for r in data[-100:]:
    r['update'] = True

db.execute(mnist_collection.insert_many(data[-100:]))
```

You can see that the inserted data, are now also populated with predictions:


```python
db.execute(mnist_collection.find_one({'update': True}))['_outputs']
```
