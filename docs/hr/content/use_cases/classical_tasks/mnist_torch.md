# MNIST classifier

## Training and Managing MNIST Predictions in SuperDuperDB

This notebook guides you through the implementation of a classic machine learning task: MNIST handwritten digit recognition. The twist? We perform the task directly in a database using SuperDuperDB.

This example makes it easy to connect any of your image recognition
model directly to your database in real-time. With SuperDuperDB, you can
skip complicated MLOps pipelines. It's a new straightforward way to
integrate your AI model with your data, ensuring simplicity, efficiency
and speed.

## Prerequisites

Before diving into the implementation, ensure that you have the
necessary libraries installed by running the following commands:

```python
!pip install superduperdb
!pip install torch torchvision matplotlib
```

## Connect to datastore

First, we need to establish a connection to a MongoDB datastore via
SuperDuperDB. You can configure the `MongoDB_URI` based on your specific
setup.

Here are some examples of MongoDB URIs:

-   For testing (default connection): `mongomock://test`
-   Local MongoDB instance: `mongodb://localhost:27017`
-   MongoDB with authentication:
    `mongodb://superduper:superduper@mongodb:27017/documents`
-   MongoDB Atlas:
    `mongodb+srv://<username>:<password>@<atlas_cluster>/<database>`

```python
from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
import os

mongodb_uri = os.getenv("MONGODB_URI","mongomock://test")

# SuperDuperDB, now handles your MongoDB database
# It just super dupers your database 
db = superduper(mongodb_uri)

# Create a collection for MNIST
mnist_collection = Collection('mnist')
```

## Load Dataset

After establishing a connection to MongoDB, the next step is to load the
MNIST dataset. SuperDuperDB's strength lies in handling diverse data
types, especially those that are challenging. To achieve this, we use an
`Encoder` in conjunction with `Document` wrappers. These components
allow Python dictionaries containing non-JSONable or bytes objects to be
seamlessly inserted into the underlying data infrastructure.

```python
import torchvision
from superduperdb.ext.pillow import pil_image
from superduperdb import Document
from superduperdb.backends.mongodb import Collection

import random

# Load MNIST images as Python objects using the Python Imaging Library.
# Each MNIST item is a tuple (image, label)
mnist_data = list(torchvision.datasets.MNIST(root='./data', download=True))

# Create a list of Document instances from the MNIST data
# Each Document has an 'img' field (encoded using the Pillow library) and a 'class' field
document_list = [Document({'img': pil_image(x[0]), 'class': x[1]}) for x in mnist_data]

# Shuffle the data and select a subset of 1000 documents
random.shuffle(document_list)
data = document_list[:1000]

# Insert the selected data into the mnist_collection which we mentioned before like: mnist_collection = Collection('mnist')
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

Following that, we build our machine learning model. SuperDuperDB
conveniently supports various frameworks, and for this example, we opt
for PyTorch, a suitable choice for computer vision tasks. In this
instance, we combine `torch` with `torchvision`.

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
        torchvision.transforms.Resize((32, 32

)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
    )(x)

# Create an instance of the LeNet-5 model
lenet_model = LeNet5(10)

# Create a SuperDuperDB model with the LeNet-5 model, preprocess, and postprocess functions
# Specify 'preferred_devices' as ('cpu',) indicating CPU preference
model = superduper(lenet_model, preprocess=preprocess, postprocess=postprocess, preferred_devices=('cpu',))
db.add(model)
```

## Train Model

Now we are ready to "train" or "fit" the model. Trainable models in
SuperDuperDB come with a sklearn-like `.fit` method.

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
    select=mnist_collection.find(), # Select the dataset from the 'mnist_collection'
    configuration=TorchTrainerConfiguration(
        identifier='my_configuration', # Unique identifier for the training configuration
        objective=cross_entropy, # The objective function (cross-entropy in this case)
        loader_kwargs={'batch_size': 10}, # DataLoader keyword arguments, batch size is set to 10
        max_iterations=10, # Maximum number of training iterations
        validation_interval=5, # Interval for validation during training
    ),
    metrics=[Metric(identifier='acc', object=lambda x, y: sum([xx == yy for xx, yy in zip(x, y)]) / len(x))], # Define a custom accuracy metric for evaluation during training
    validation_sets=[
        # Define a validation dataset using a subset of data with '_fold' equal to 'valid'
        Dataset(
            identifier='my_valid',
            select=Collection('mnist').find({'_fold': 'valid'}),
        )
    ],
    distributed=False, # Set to True if distributed training is enabled
)
```

## Monitoring Training Efficiency

You can monitor the training efficiency with visualization tools like
Matplotlib:

```python
from matplotlib import pyplot as plt

# Load the model from the database
model = db.load('model', model.identifier)

# Plot the accuracy values
plt.plot(model.metric_values['my_valid/acc'])
plt.show()
```

## On-the-fly Predictions

After training the model, you can continuously predict on new data as it arrives. By activating a `listener` for the database, the model can make predictions on incoming data changes without having to load all the data client-side. The listen toggle triggers the model to predict based on updates in the incoming data.

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
# Execute find_one() to retrieve a single document from the 'mnist_collection'. 
r = db.execute(mnist_collection.find_one({'_fold': 'valid'}))

# Unpack the document and extract its content
r.unpack()
```

## Verification

The models "activated" can be seen here:

```python
# Show the status of the listener
db.show('listener')
```

We can verify that the model is activated, by inserting the rest of the
data:

```python
# Iterate over the last 100 elements in the 'data' list
for r in data[-100:]:
    # Update the 'update' field to True for each document
    r['update'] = True

# Insert the updated documents (with 'update' set to True) into the 'mnist_collection'
db.execute(mnist_collection.insert_many(data[-100:]))
```

You can see that the inserted data, are now also populated with
predictions:

```python
# Execute find_one() to retrieve a single sample document from 'mnist_collection'
# where the 'update' field is True
sample_document = db.execute(mnist_collection.find_one({'update': True}))['_outputs']

# A sample document 
print(sample_document)
```
