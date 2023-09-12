# Transfer learning using Sentence Transformers and Scikit-Learn

In this example, we'll be demonstrating how to simply implement transfer learning using SuperDuperDB.
You'll find related examples on vector-search and simple training examples using scikit-learn in the 
the notebooks directory of the project. Transfer learning leverages similar components, and may be used synergistically with vector-search. Vectors are, after all, simultaneously featurizations of 
data and may be used in downstream learning tasks.

Let's first connect to MongoDB via SuperDuperDB, you read explanations of how to do this in 
the docs, and in the `notebooks/` directory.


```python
from superduperdb import superduper
from superduperdb.db.mongodb.query import Collection
import pymongo

db = superduper(
    pymongo.MongoClient().documents
)

collection = Collection('transfer')
```

We'll use textual data labelled with sentiment, to test the functionality. Transfer learning 
can be used on any data which can be processed with SuperDuperDB models.


```python
import numpy
from datasets import load_dataset

from superduperdb.container.document import Document as D

data = load_dataset("imdb")

train_data = [
    D({'_fold': 'train', **data['train'][int(i)]}) 
    for i in numpy.random.permutation(len(data['train']))
][:5000]

valid_data = [
    D({'_fold': 'valid', **data['test'][int(i)]}) 
    for i in numpy.random.permutation(len(data['test']))
][:500]

db.execute(collection.insert_many(train_data))

r = db.execute(collection.find_one())
r
```

Let's create a SuperDuperDB model based on a `sentence_transformers` model.
You'll notice that we don't necessarily need a native SuperDuperDB integration to a model library 
in order to leverage its power with SuperDuperDB. For example, in this case, we just need 
to configure the `Model` wrapper to interoperate correctly with the `SentenceTransformer` class. After doing this, we can link the model to a collection, and /docs/docs/usage/models#daemonizing-models-with-listeners the model using the `listen=True` keyword:


```python
from superduperdb.container.model import Model
import sentence_transformers

from superduperdb.ext.numpy.array import array

m = Model(
    identifier='all-MiniLM-L6-v2',
    object=sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2'),
    encoder=array('float32', shape=(384,)),
    predict_method='encode',
    batch_predict=True,
)

m.predict(
    X='text',
    db=db,
    select=collection.find(),
    listen=True
)
```

Now that we've created and added the model which computes features for the `"text"`, we can train a 
downstream model using Scikit-Learn:


```python
from sklearn.svm import SVC

model = superduper(
    SVC(gamma='scale', class_weight='balanced', C=100, verbose=True),
    postprocess=lambda x: int(x)
)

model.fit(
    X='text',
    y='label',
    db=db,
    select=collection.find().featurize({'text': 'all-MiniLM-L6-v2'}),
)
```

Now that the model has been trained, we can apply the model to the database, also daemonizing the model 
with `listen=True`.


```python
model.predict(
    X='text',
    db=db,
    select=collection.find().featurize({'text': 'all-MiniLM-L6-v2'}),
    listen=True,
)
```

To verify that this process has worked, we can sample a few records, to inspect the sanity of the predictions


```python
r = next(db.execute(collection.aggregate([{'$sample': {'size': 1}}])))
print(r['text'][:100])
print(r['_outputs']['text']['svc'])
```
