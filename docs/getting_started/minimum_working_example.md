# Minimum working example

To check that everything is working correctly cut and paste this code into a Jupyter notebook:

```python
import numpy as np
from pymongo import MongoClient
from superduperdb.container.document import Document as D
from superduperdb.ext.numpy.array import array
from superduperdb.db.mongodb.query import Collection, InsertMany
import superduperdb as s

db = s.superduper(MongoClient().documents)
collection = Collection(name='docs')

a = array('float64', shape=(32,))

db.execute(
    collection.insert_many([
        D({'x': a(np.random.randn(32))})
        for _ in range(100)
    ], encoders=(a,))
)

model = s.core.model.Model(
    identifier='test-model',
    object=lambda x: x + 1,
    encoder=a,
)

model.predict(X='x', db=db, select=collection.find())

print(db.execute(collection.find_one()))
```

### Explanation

1. We wrap the `pymongo` database connector with the `superduper` decorator, allowing SuperDuperDB to communicate with MongoDB and install AI into the database.
2. We insert several `numpy` arrays, using the encoder `a` to encode these as `bytes` in the database.
3. We wrap our model, which in this case, is a simple `lambda` function.
4. We apply the model to store predictions on the inserted data in the database.
5. We query the database, to retrieve a sample datapoint.