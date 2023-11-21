# Sentiment analysis with transformers


```python
!pip install superduperdb==0.0.12
!pip install datasets
```

In this notebook we implement a classic NLP use-case using Hugging Face's `transformers` library.
We show that this use-case may be implementing directly in SuperDuperDB using MongoDB as the
data-backend. 


```python
from datasets import load_dataset
import numpy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from superduperdb import Document as D, Dataset 
from superduperdb.ext.transformers import TransformersTrainerConfiguration, Pipeline
```

SuperDuperDB supports MongoDB as a databackend.
Correspondingly, we'll import the python MongoDB client pymongo and "wrap" our database to convert it 
to a SuperDuper Datalayer:


```python
import os
from superduperdb.backends.mongodb import Collection

# Uncomment one of the following lines to use a bespoke MongoDB deployment
# For testing the default connection is to mongomock

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
mongodb_uri = os.getenv("MONGODB_URI","mongomock://test")
# mongodb_uri = "mongodb://localhost:27017"
# mongodb_uri = "mongodb://superduper:superduper@mongodb:27017/documents"
# mongodb_uri = "mongodb://<user>:<pass>@<mongo_cluster>/<database>"
# mongodb_uri = "mongodb+srv://<username>:<password>@<atlas_cluster>/<database>"

# Super-Duper your Database!
from superduperdb import superduper
db = superduper(mongodb_uri)
collection = Collection('imdb')
```

We use the IMDB dataset for training the model:


```python
data = load_dataset("imdb")

# increase this number to do serious training
N_DATAPOINTS = 4

db.execute(collection.insert_many([
    D({'_fold': 'train', **data['train'][int(i)]}) for i in numpy.random.permutation(len(data['train']))[:N_DATAPOINTS]
]))

db.execute(collection.insert_many([
    D({'_fold': 'valid', **data['test'][int(i)]}) for i in numpy.random.permutation(len(data['test']))[:N_DATAPOINTS]
]))
```

Check a sample from the database:


```python
r = db.execute(collection.find_one())
r
```

Create a tokenizer and use it to provide a data-collator for batching inputs:


```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model = Pipeline(
    identifier='my-sentiment-analysis',
    task='text-classification',
    preprocess=tokenizer,
    object=model,
    preprocess_kwargs={'truncation': True},
)
```


```python
model.predict('This is another test', one=True)
```

We'll evaluate the model using a simple accuracy metric. This metric gets logged in the
model's metadata during training:


```python
training_args = TransformersTrainerConfiguration(
    identifier='sentiment-analysis',
    output_dir='sentiment-analysis',
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch",
    use_cpu=True,
    evaluation_strategy='epoch',
    do_eval=True,
)
```

Now we're ready to train the model:


```python
from superduperdb import Metric

model.fit(
    X='text',
    y='label',
    db=db,
    select=collection.find(),
    configuration=training_args,
    validation_sets=[
        Dataset(
            identifier='my-eval',
            select=collection.find({'_fold': 'valid'}),
        )
    ],
    data_prefetch=False,
    metrics=[Metric(
        identifier='acc',
        object=lambda x, y: sum([xx == yy for xx, yy in zip(x, y)]) / len(x)
    )]
)                                                                            
```

We can verify that the model gives us reasonable predictions:


```python
model.predict("This movie sucks!", one=True)
```
