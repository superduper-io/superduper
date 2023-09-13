---
sidebar_position: 1
---

# Welcome to the SuperDuperDB Docs

## What is SuperDuperDB?

SuperDuperDB is a Python package which provides tools for developers to apply AI and machine learning in their already deployed datastore, and simultaneously to set-up a scalable, open-source and auditable environment to do this.

![](/img/SuperDuperDB_diagram.svg)

### What can you do with SuperDuperDB?

- **Deploy** all your AI models to automatically **compute outputs (inference)** in your datastore in a single environment with simple Python commands.  
- **Train** models on your data in your datastore simply by querying without additional ingestion and pre-processing.  
- **Integrate** AI APIs (such as OpenAI) to work together with other models on your data effortlessly. 
- **Search** your data with vector-search, including model management and serving.

 ### Why choose SuperDuperDB?

- Avoid data duplication, pipelines and duplicate infrastructure with a **single scalable deployment**.
- **Deployment always up-to-date** as new data is handled automatically and immediately.
- A **simple and familiar Python interface** that can handle even the most complex AI use-cases.

### Who is SuperDuperDB for?

  - **Python developers** using datastores (databases/ lakes/ warehouses) who want to build AI into their applications easily.
  - **Data scientists & ML engineers** who want to develop AI models using their favourite tools, with minimum infrastructural overhead.
  - **Infrastructure engineers** who want a single scalable setup that supports both local, on-prem and cloud deployment.

### SuperDuperDB transforms your datastore into:

  - **An end-to-end live AI deployment** which includes a **model repository and registry**, **model training** and **computation of outputs/ inference** 
  - **A feature store** in which the model outputs are stored alongside the inputs in any data format. 
  - **A fully functional vector database** to easily generate vector embeddings of your data with your favorite models and APIs and connect them with your datastore (and/ or) vector database.

## Code snippets to pique your interest!

### Model frameworks directly integrated with the datastore 

SuperDuperDB includes wrappers for treating models from diverse AI frameworks across the open-source Python ecosystem uniformly in combination with the datastore, using a scikit-learn-like
`.fit()` and `.predict()` API.

```python
# [ Code snippets assume access to a running datastore instance read/write ]
from sklearn.svm import SVC
import pymongo

from superduperdb.db.mongodb import Collection
from superduperdb import superduper

# Models and datastore clients can be converted to SuperDuperDB objects with a simple wrapper.
model = superduper(SVC())

# SuperDuperDB uses MongoDB by default. SQL integrations are on the way.
db = superduper(pymongo.MongoClient().my_db)

# Once wrapped, we can fit and predict "in" the datastore, simply
# specifying the data to be processed with a query.
coll = Collection(name='my_collection')
model.fit(X='input_col', y='predict_col', db=db, select=coll.find({'_fold': 'train'}))

# Predictions are saved in the datastore alongside the inputs.
model.predict(X='input_col', db=db, select=coll.find({'_fold': 'valid'}))
```

### Continuous model processing on incoming data

SuperDuperDB contains components allowing developers to configure models to continuously infer outputs on specified data, and save the outputs back to the datastore.

```python
# listen the datastore for incoming data, and process this with a model
# Model outputs are continuously stored in the input records
model.predict(X='input_col', db=db, select=coll.find(), listen=True)
```

### Use models outputs as inputs to downstream models

Simply add a simple method `featurize` to your queries, to register the fact that one model depends on another:

```python
model.predict(
    X='input_col',
    db=db,
    select=coll.find().featurize({'X': '<upstream-model-id>'}),  # already registered upstream model-id
    listen=True,
)
```

### Support for "tricky" datatypes

SuperDuperDB includes tools for working with the datastore using the complex data types necessary for AI, such as vectors, tensors, images, audio etc. Native python types may be flexibly saved to the DB, to ease use in tricky AI use-cases, such as computer vision:

```python
from superduperdb.ext.pillow import pil_image as i

# Encoders are first class SuperDuperDB objects which deal with serializing
# "non-standard" data to the datastore 
db.execute(
    coll.insert_many([
    {'img': i(PIL.image.open(path))} for path in images
    ])
)
```

Data may be reloaded using standard datastore queries, and conveniently reused in downstream applications, or consumed as direct inputs to AI models. The data loaded are instances of the same Python classes as inserted.

```python
r = db.execute(coll.find_one())
r['img'].x
<PIL.PngImagePlugin.PngImageFile image mode=RGBA size=1164x860>
```

### Use your classical datastore as a vector database

SuperDuperDB contains functionality allowing users to treat their standard datastore as a vector-search database, integrating your primary datastore with key-players in the open-source vector-search space.
```python
# First a "listener" makes sure vectors stay up-to-date
indexing_listener = listener(model=OpenAIEmbedding(), key='text', select=collection.find())

# This "listener" is linked with a "VectorIndex"
db.add(VectorIndex('my-index', indexing_listener=indexing_listener))

# The "VectorIndex" may be used to search data. Items to be searched against are passed
# to the registered model and vectorized. No additional app layer is required.
# By default, SuperDuperDB uses LanceDB for vector comparison operations
db.execute(collection.like({'text': 'clothing item'}, 'my-index').find({'brand': 'Nike'}))
```