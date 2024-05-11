# Training models

## Introduction

This notebook guides you through the implementation of a classic machine learning task: MNIST handwritten digit recognition. The twist? We perform the task directly on data hosted in a database using SuperDuperDB.

This example makes it easy to connect any of your image recognition model directly to your database in real-time. With SuperDuperDB, you can skip complicated MLOps pipelines. It's a new straightforward way to integrate your AI model with your data, ensuring simplicity, efficiency and speed. 


```python
!pip install torch torchvision
```

## Connect to datastore 

First, we need to establish a connection to a MongoDB datastore via SuperDuperDB. You can configure the `MongoDB_URI` based on your specific setup. 


```python
from superduperdb import superduper

db = superduper('mongomock://')
```

## Load Dataset

After establishing a connection to MongoDB, the next step is to load the MNIST dataset. SuperDuperDB's strength lies in handling diverse data types, especially those that are challenging. To achieve this, we use an `Encoder` in conjunction with `Document` wrappers. These components allow Python dictionaries containing non-JSONable or bytes objects to be seamlessly inserted into the underlying data infrastructure.


```python
import torchvision
from superduperdb.ext.pillow import pil_image
from superduperdb import Document
from superduperdb.backends.mongodb import Collection

import random

# Load MNIST images as Python objects using the Python Imaging Library.
# Each MNIST item is a tuple (image, label)
mnist_data = list(torchvision.datasets.MNIST(root='./data', download=True))

document_list = [{'img': x[0], 'class': x[1]}) for x in mnist_data]

# Shuffle the data and select a subset of 1000 documents
random.shuffle(document_list)
data = document_list[:1000]

# Insert the selected data into the mnist_collection which we mentioned before like: mnist_collection = Collection('mnist')
db['mnist'].insert_many(data[:-100]).execute()
```

Now that the images and their classes are inserted into the database, we can query the data in its original format. Particularly, we can use the `PIL.Image` instances to inspect the data.


```python
# Get and display one of the images
r = db['mnist'].find_one().execute()
r.unpack()['img']
```

## Build Model

Following that, we build our machine learning model. SuperDuperDB conveniently supports various frameworks, and for this example, we opt for PyTorch, a suitable choice for computer vision tasks. In this instance, we combine `torch` with `torchvision`.

To facilitate communication with the SuperDuperDB `Datalayer`, we design `postprocess` and `preprocess` functions. These functions are then encapsulated with the model, preprocessing, and postprocessing steps to create a native SuperDuperDB handler.


```python
import torch

# Define the LeNet-5 architecture for image classification
class LeNet5(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Layer 1
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # Layer 2
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # Fully connected layers
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

# Postprocess function for the model output    
def postprocess(x):
    return int(x.topk(1)[1].item())

# Preprocess function for input data
def preprocess(x):
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
    )(x)

# Create an instance of the LeNet-5 model
lenet_model = LeNet5(10)

# Create a SuperDuperDB model with the LeNet-5 model, preprocess, and postprocess functions
# Specify 'preferred_devices' as ('cpu',) indicating CPU preference
model = TorchModel(
    identifier='my-model',
    object=lenet_model,
    preprocess=preprocess,
    postprocess=postprocess, 
    preferred_devices=('cpu',),
)

# Check that the model successfully creates predictions over single data-points
print(model.predict_one(data[0]['img']))
```

## Train Model

Now we are ready to "train" or "fit" the model. Trainable models in SuperDuperDB come with a sklearn-like `.fit` method. 



```python
from torch.nn.functional import cross_entropy

from superduperdb import Metric
from superduperdb import Dataset
from superduperdb.ext.torch.model import TorchTrainer

acc = lambda x, y: sum([xx == yy for xx, yy in zip(x, y)]) / len(x))

accuracy = Metric(identifier='acc', object=acc)

model.validation = Validation(
    datasets=[
        Dataset(
            identifier='my-valid',
            select=db['mnist'].find({'_fold': 'valid'})
        )
    ],
    metrics=[accuracy],
)

model.trainer = TorchTrainer(
    identifier='my-trainer',
    objective=cross_entropy,
    loader_kwargs={'batch_size': 10},
    max_iterations=10,
    validation_interval=5,
    select=mnist_collection.find(),
    key=('img', 'class'),
)

db.apply(model)
```


```python
from matplotlib import pyplot as plt

# Load the model from the database
model = db.load('model', model.identifier)

# Plot the accuracy values
plt.plot(model.trainer.metric_values['my-valid/acc'])
plt.show()
```
