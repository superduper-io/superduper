
# Image Feature-Store Using Torchvision on MongoDB

```python
!pip install superduperdb
!pip install torchvision
```

In this example, we show how to utilize a pre-trained network from `torchvision` to produce image features. The images are automatically fetched and stored in MongoDB. We use a subset of the CoCo dataset (https://cocodataset.org/#home) to illustrate the process.

Real-life use cases for creating a database of image features using a pre-trained network in `torchvision`:

1. **Image Search and Retrieval:**
   - **Use Case:** Enhance image search capabilities in e-commerce platforms.
   - **How:** Generate image features for products using a pre-trained network. Store these features in a database for efficient image retrieval, making it easier for users to find similar products.

2. **Content-Based Recommendation Systems:**
   - **Use Case:** Improve content recommendations in media streaming services.
   - **How:** Extract image features from movie or show frames. Store these features in a database to recommend content with similar visual characteristics to users based on their preferences.

3. **Facial Recognition in Security Systems:**
   - **Use Case:** Strengthen facial recognition systems in security applications.
   - **How:** Utilize a pre-trained neural network to extract facial features from images. Store these features in a database for quick and accurate identification in security and surveillance scenarios.

4. **Medical Image Analysis:**
   - **Use Case:** Assist in medical diagnostics through image analysis.
   - **How:** Extract features from medical images (X-rays, MRIs, etc.) using a pre-trained network. Store these features to aid in the development of diagnostic tools or systems for healthcare professionals.

5. **Automated Image Tagging:**
   - **Use Case:** Streamline image organization in photo libraries or social media platforms.
   - **How:** Extract features from uploaded images using a pre-trained model. Use these features to automatically generate relevant tags, making it easier for users to search and categorize their photos.

These use cases demonstrate how creating a database of image features using `torchvision` can be applied across various domains to enhance functionality and improve user experiences. Guess what, all can be done with `superduperdb` like this example.

```python
# Download the zip file
!curl -O https://superduperdb-public.s3.eu-west-1.amazonaws.com/valsmall2014.zip

# Unzip the contents of the zip file (assuming the file is already downloaded)
!unzip -qq valsmall2014.zip
```

## Connect to Datastore

First, we need to establish a connection to a MongoDB datastore via SuperDuperDB. You can configure the `MongoDB_URI` based on your specific setup.

Here are some examples of MongoDB URIs:

- For testing (default connection): `mongomock://test`
- Local MongoDB instance: `mongodb://localhost:27017`
- MongoDB with authentication: `mongodb://superduper:superduper@mongodb:27017/documents`
- MongoDB Atlas: `mongodb+srv://<username>:<password>@<atlas_cluster>/<database>`

```python
import os
from superduperdb import superduper
from superduperdb.backends.mongodb import Collection

# Get the MongoDB URI from the environment variable or use a default value
mongodb_uri = os.getenv("MONGODB_URI", "mongomock://test")

# SuperDuperDB, now handles your MongoDB database
# It just super dupers your database
db = superduper(mongodb_uri)

# Specify a collection named 'coco'
collection = Collection('coco')
```

Next, we include all image URIs in MongoDB. These URIs may include a mix of local file paths (`file://...`), web URLs (`http...`), and S3 URIs (`s3://...`). Once the URIs are added, SuperDuperDB automatically loads their content into MongoDB without the need for extra overhead or job definitions.

```python
import glob
import random

from superduperdb import Document as D
from superduperdb.ext.pillow import pil_image as i

# Get a list of file URIs for all JPEG images in the 'valsmall2014' directory
uris = [f'file://{x}' for x in glob.glob('valsmall2014/*.jpg')]

# Insert documents into the 'coco' collection in the MongoDB database
db.execute(collection.insert_many([D({'img': i(uri=uri)}) for uri in uris], encoders=(i,)))  # Here the image is encoded with pillow
```

To confirm the correct storage of images in the `Datalayer`, we can perform a verification check.

```python
# Import the display function from the IPython.display module
from IPython.display import display

# Define a lambda function for displaying images with resizing to avoid potential Jupyter crashes
display_image = lambda x: display(x.resize((round(x.size[0] * 0.5), round(x.size[1] * 0.5))))

# Retrieve the 'img' attribute from the result of collection.find_one() using db.execute()
# Note: This assumes that db is an instance of a database connection wrapped with superduperdb
x = db.execute(collection.find_one())['img'].x

# Display the image using the previously defined lambda function
display_image(x)
```

Let's build the `torch` + `torchvision` model using the `TorchModel` wrapper from SuperDuperDB. This allows for the incorporation of custom pre- and post-processing steps along with the model's forward pass.

```python
# Import necessary libraries and modules from torchvision and torch
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models

import warnings

# Import custom modules
from superduperdb.ext.torch import TorchModel, tensor

# Define a series of image transformations using torchvision.transforms.Compose
t = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize the input image to 224x224 pixels (must same as here)
    transforms.CenterCrop((224, 224)),  # Perform a center crop on the resized image
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the tensor with specified mean and standard deviation
])

# Define a preprocess function that applies the defined transformations to an input image
def preprocess(x):
    try:
        return t(x)
    except Exception as e:
        # If an exception occurs during preprocessing, issue a warning and return a tensor of zeros
        warnings.warn(str(e))
        return torch.zeros(3, 224, 224)

# Load the pre-trained ResNet-50 model from torchvision
resnet50 = models.resnet50(pretrained=True)

# Extract all layers of the ResNet-50 model except the last one
modules = list(resnet50.children())[:-1]
resnet50 = nn.Sequential(*modules)

# Create a TorchModel instance with the Res

Net-50 model, preprocessing function, and postprocessing lambda
model = TorchModel(
    identifier='resnet50',
    preprocess=preprocess,
    object=resnet50,
    postprocess=lambda x: x[:, 0, 0],  # Postprocess by extracting the top-left element of the output tensor
    encoder=tensor(torch.float, shape=(2048,))  # Specify the encoder configuration
)
```

To ensure the correctness of the `model`, let's test it on a single data point by setting `one=True`.

```python
# Assuming x is an input tensor, you're making a prediction using the configured model
# with the one=True parameter specifying that you expect a single prediction result.
model.predict(x, one=True)
```

Now that the model is prepared, we can apply it to the images stored in the `Datalayer`.

```python
# Assuming X is the input data, in this case, images ('img')
prediction_results = model.predict(
    X='img',                # Specify the input data (images)
    db=db,                  # Provide the database connection or object
    select=collection.find(),  # Specify the data to be used for prediction (fetch all data from the collection)
    batch_size=10,          # Set the batch size for making predictions
    max_chunk_size=3000,    # Set the maximum size of data chunks processed at once
    in_memory=False,        # Indicate that the data is not loaded entirely into memory, processed in chunks
    listen=True             # Enable listening mode, suggesting real-time or asynchronous prediction
)
```

To confirm that the features were stored in the `Datalayer`, you can examine them in the `_outputs.img.resnet50` field.

```python
# Execute find_one() to retrieve a single document from the collection.
result = db.execute(collection.find_one())

# The purpose of unpack() is to extract or process the data
result.unpack()
```
