# Sentiment Analysis Using transformers on MongoDB

```python
!pip install superduperdb
!pip install datasets
```

In this document, we're doing sentiment analysis using Hugging Face's `transformers` library. We demonstrate that you can perform this task seamlessly in SuperDuperDB, using MongoDB to store the data.

Sentiment analysis has some real-life use cases:

1. **Customer Feedback & Review Analysis:** Analyzing customer reviews and feedback to understand overall satisfaction, identify areas for improvement, and respond to customer concerns. It is used in the E-commerce industry frequently.

2. **Brand Monitoring:** Monitoring social media, blogs, news articles, and online forums to gauge public sentiment towards a brand, product, or service. Addressing negative sentiment and capitalizing on positive feedback.

Sentiment analysis plays a crucial role in understanding and responding to opinions and emotions expressed across various domains, contributing to better decision-making and enhanced user experiences.

All of these can be done with your `existing database` and `SuperDuperDB`. You can integrate similar code into your ETL infrastructure as well. Let's see an example.

```python
from datasets import load_dataset
import numpy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import Document (aliased as D) and Dataset from the superduperdb module
from superduperdb import Document as D, Dataset
```

## Connect to datastore

SuperDuperDB can work with MongoDB (one of many supported databases) as its database backend. To make this connection, we'll use the Python MongoDB client, pymongo, and "wrap" our database to transform it into a SuperDuper Datalayer.

First, we need to establish a connection to a MongoDB datastore via SuperDuperDB. You can configure the `MongoDB_URI` based on your specific setup.

Here are some examples of MongoDB URIs:

- For testing (default connection): `mongomock://test`
- Local MongoDB instance: `mongodb://localhost:27017`
- MongoDB with authentication: `mongodb://superduper:superduper@mongodb:27017/documents`
- MongoDB Atlas: `mongodb+srv://<username>:<password>@<atlas_cluster>/<database>`

```python
import os
from superduperdb.backends.mongodb import Collection
from superduperdb import superduper

# Set an environment variable to enable PyTorch MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Get the MongoDB URI from the environment variable "MONGODB_URI," defaulting to "mongomock://test"
mongodb_uri = os.getenv("MONGODB_URI","mongomock://test")

# SuperDuperDB, now handles your MongoDB database
# It just super dupers your database 
db = superduper(mongodb_uri)

# Collection instance named 'imdb' in the database
collection = Collection('imdb')
```

We train the model using the IMDB dataset.

```python
# Load the IMDb dataset using the load_dataset function from the datasets module
data = load_dataset("imdb")

# Set the number of datapoints to be used for training and validation. Increase this number to do serious training
N_DATAPOINTS = 4

# Insert randomly selected training datapoints into the 'imdb' collection in the database
db.execute(collection.insert_many([
    # Insert training data into your database from the dataset. Create Document instances for each training datapoint, setting '_fold' to 'train'
    D({'_fold': 'train', **data['train'][int(i)]}) for i in numpy.random.permutation(len(data['train']))[:N_DATAPOINTS]
]))

# Insert randomly selected validation datapoints into the 'imdb' collection in the database
db.execute(collection.insert_many([
    # Insert validation data into your database from the dataset. Create Document instances for validation datapoint, setting '_fold' to 'valid'
    D({'_fold': 'valid', **data['test'][int(i)]}) for i in numpy.random.permutation(len(data['test']))[:N_DATAPOINTS]
]))
```

Retrieve a sample from the database.

```python
# Execute the find_one() method to retrieve a single document from the 'imdb' collection. To check if the database insertion is done okay.
r = db.execute(collection.find_one())
r
```

Build a tokenizer and utilize it to create a data collator for batching inputs.

```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Instantiate a sequence classification model for the 'distilbert-base-uncased' model with 2 labels
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Create a Pipeline instance for sentiment analysis
# identifier: A unique identifier for the pipeline
# task: The type of task the pipeline is designed for, in this case, 'text-classification'
# preprocess: The tokenizer to use for preprocessing
# object: The model for text classification
# preprocess_kwargs: Additional keyword arguments for the tokenizer, e.g., truncation
model = Pipeline(
    identifier='my-sentiment-analysis',
    task='text-classification',
    preprocess=tokenizer,
    object=model,
    preprocess_kwargs={'truncation': True},
)
```

```python
# Assuming 'This is another test' is the input text for prediction
# You're making a prediction using the configured pipeline model
# with the one=True parameter specifying that you expect a single prediction result.
model.predict('This is another test', one=True)
```

We'll assess the model using a straightforward accuracy metric. This metric will be recorded in the model's metadata as part of the training process.

```python
# Import TransformersTrainerConfiguration from the superduperdb.ext.transformers module
from superduperdb.ext.transformers import TransformersTrainerConfiguration

# Create a configuration for training a transformer model
training_args = TransformersTrainerConfiguration(
    identifier='sentiment-analysis',  # A unique identifier for the training configuration
    output_dir='sentiment-analysis',  # The directory where model predictions will be saved
    learning_rate=2e-5,  # The learning rate for training the model
    per_device_train_batch_size=2,  # Batch size per GPU (or CPU) for training
    per_device_eval_batch_size=2,  # Batch size per GPU (or CPU) for evaluation
    num_train_epochs=2,  # The number of training epochs
    weight_decay=0.01,  # Weight decay for regularization
    save_strategy="epoch",  # Save model checkpoints after each epoch
    use_cpu=True,  # Use CPU for training (set to False if you want to use GPU)
    evaluation_strategy='epoch',  # Evaluate the model after each epoch
    do_eval=True,  # Perform evaluation during training
)
```

Now we're ready to train the model:

```python
# Import the Metric class from the superduperdb module
from superduperdb import Metric

# Fit the model using training data and specified configuration
model.fit(
    X='text',  # Input data (text)
    y='label',  # Target variable (label)
    db=db,

  # Super Duper wrapped Database connection
    select=collection.find(),  # Specify the data to be used for training (fetch all data from the collection)
    configuration=training_args,  # Training configuration using the previously defined TransformersTrainerConfiguration
    validation_sets=[
        # Define a validation dataset using a subset of data with '_fold' equal to 'valid'
        Dataset(
            identifier='my-eval',
            select=collection.find({'_fold': 'valid'}),
        )
    ],
    data_prefetch=False,  # Disable data prefetching during training
    metrics=[
        # Define a custom accuracy metric for evaluation during training
        Metric(
            identifier='acc',
            object=lambda x, y: sum([xx == yy for xx, yy in zip(x, y)]) / len(x)
        )
    ]
)
```

We can confirm that the model produces sensible predictions by examining the output. If you are okay with the performance, you may predict it on your whole database and save it for future reference. All can be done on SuperDuperDB in real-time.

```python
# Assuming "This movie sucks!" is the input text for sentiment analysis
# You're making a prediction using the configured pipeline model
# with the one=True parameter specifying that you expect a single prediction result.
sentiment_prediction = model.predict("This movie sucks!", one=True)

print("Sentiment Prediction", sentiment_prediction)
```
