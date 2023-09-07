# MNIST using scikit-learn and SuperDuperDB

In a [previous example](mnist_torch.html) we discussed how to implement MNIST classification with CNNs in `torch`
using SuperDuperDB. 


```python
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score,classification_report
import numpy as np
from sklearn import svm
```

As before we'll import the python MongoDB client `pymongo`
and "wrap" our database to convert it to a SuperDuper `Datalayer`:


```python
import pymongo
from superduperdb import superduper

db = pymongo.MongoClient().documents

db = superduper(db)
```

Similarly to last time, we can add data to SuperDuperDB in a way which very similar to using `pymongo`.
This time, we'll add the data as `numpy.array` to SuperDuperDB, using the `Document-Encoder` formalism:


```python
from superduperdb.ext.numpy.array import array
from superduperdb.container.document import Document as D
from superduperdb.db.mongodb.query import Collection

mnist = fetch_openml('mnist_784')
ix = np.random.permutation(10000)
X = np.array(mnist.data)[ix, :]
y = np.array(mnist.target)[ix].astype(int)

a = array('float64', shape=(784,))

collection = Collection(name='mnist')

data = [D({'img': a(X[i]), 'class': int(y[i])}) for i in range(len(X))]

db.execute(
    collection.insert_many(data, encoders=[a])
)
```


```python
db.execute(collection.find_one())
```

Models are built similarly to the `Datalayer`, by wrapping a standard Python-AI-ecosystem model:


```python
model = superduper(
    svm.SVC(gamma='scale', class_weight='balanced', C=100, verbose=True),
    postprocess=lambda x: int(x)
)
```

Now let's fit the model. The optimization uses Scikit-Learn's inbuilt training procedures.
Unlike in a standard `sklearn` use-case, we don't need to fetch the data client side. Instead, 
we simply name the fields in the MongoDB collection which we'd like to use.


```python
model.fit(X='img', y='class', db=db, select=collection.find())
```

Installed models and functionality can be viewed using `db.show`:


```python
db.show('model')
```

The model may be reloaded in another session from the database. 
As with `.fit`, the model may be applied to data in the database with `.predict`:


```python
m = db.load('model', 'svc')
m.predict(X='img', db=db, select=collection.find(), max_chunk_size=3000)
```

We can verify that the predictions make sense by fetching a few random data-points:


```python
r = next(db.execute(collection.aggregate([{'$match': {'_fold': 'valid'}} ,{'$sample': {'size': 1}}])))
print(r['class'])
print(r['_outputs'])
```
