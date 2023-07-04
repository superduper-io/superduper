# SuperDuperDB documentation

🚀 Welcome to SuperDuperDB! 🚀

## Mission

SuperDuperDB is an open-source project, whose primary goal is to smoothen the developer journey
between data and AI models.

- An easy-to-use, extensible and comprehensive python framework for integrating AI and 
  ML directly to the datalayer.
  
- To empower developers, data scientists and architects to leverage the vast PyData, python AI
  and open-source ecosystem in their datalayer deployments.
  
- Provide ways-of-working with data which enable scalability and industrial scale deployment,
  as well as providing easy-to-use tools for the individual developer.
  
- Open-source (Apache 2.0) and prioritizing open-source integrations in our roadmap going forward

- Enable individuals and organizations to circumvent vendor lock-in strategies now ubiquitous
  in the AI and ML landscapes, by providing a clear toolset to flexibly deploy AI at the 
  datalayer without necessitating subscriptions, cloud installations, OpenAI functionality 

## Features

### Model frameworks directly integrated with databases

SuperDuperDB includes wrappers for treating models from diverse AI frameworks across the open-source Python ecosystem uniformly in combination with databases, using a scikit-learn-like
`.fit` and `.predict` API.

```python
# [ Code vignettes assume access to a running MongoDB instance read/write ]
from sklearn.svm import SVC
import pymongo

from superduperdb.datalayer.mongodb import Collection
from superduperdb import superduper

# Models and database clients can be converted to SuperDuperDB objects with a simple wrapper.
model = superduper(SVC())

# SuperDuperDB uses MongoDB by default. SQL integrations are on the way.
db = superduper(pymongo.MongoClient().my_db)

# Once wrapped, we can fit and predict "in" the database, simply
# specifying the data to be processed with a query.
coll = Collection(name='my_collection')
model.fit(X='input_col', y='predict_col', db=db, select=coll.find({'_fold': 'train'}))

# Predictions are saved in the database alongside the inputs.
model.predict(X='input_col', db=db, select=coll.find({'_fold': 'valid'}))
```

### Support for "tricky" datatypes

SuperDuperDB includes tools for working in the database with the complex data types necessary for AI, such as vectors, tensors, images, audio etc. Native python types may be flexibly saved to the DB, to ease use in tricky AI use-cases, such as computer vision.

```python
from superduperdb.encoders.pillow import pil_image as i

# Encoders are first class SuperDuperDB objects which deal with serializing
# "non-standard" data to the database
db.execute(
    coll.insert_many([
        {'img': i(PIL.image.open(path))} for path in images
    ])
)
```

Data may be reloaded using standard database queries, and conveniently reused in downstream applications, or consumed as direct inputs to AI models.
The data loaded are instances of the same Python classes as inserted.

```python
>>> r = db.execute(coll.find_one())
>>> r['img'].x
<PIL.PngImagePlugin.PngImageFile image mode=RGBA size=1164x860> 
```

### Continuous model processing on incoming data
SuperDuperDB contains components allowing developers to configure models to continuously infer outputs on specified data, and save the outputs back to the database.

```python
# Watch the database for incoming data, and process this with a model
# Model outputs are continuously stored in the input records
db.add(
    Watcher(
    	model=my_model,       # model which processes data
    	key='X',              # key/ field/ column as model input
    	select=collection.find({'img': {'$exists': 1}})        # data which should be processed
    )
)
```

### Use your classical database as a vector database

SuperDuperDB contains functionality allowing users to treat their standard database as a vector-search database, integrating your primary database with key-players in the open-source vector-search space.
```python
# First a "Watcher" makes sure vectors stay up-to-date
indexing_watcher = Watcher(model=OpenAIEmbedding(), key='text', select=collection.find())

# This "Watcher" is linked with a "VectorIndex"
db.add(VectorIndex('my-index', indexing_watcher=indexing_watcher))

# The "VectorIndex" may be used to search data. Items to be searched against are passed 
# to the registered model and vectorized. No additional app layer is required.
# By default, SuperDuperDB uses LanceDB for vector comparison operations
db.execute(collection.like({'text': 'clothing item'}, 'my-index').find({'brand': 'Nike'}))
```

## Contents

```{toctree}
:maxdepth: 2

common_issues

```