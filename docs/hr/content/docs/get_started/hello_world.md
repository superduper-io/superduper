---
sidebar_position: 5
tags:
  - quickstart
---

# Hello world 

To check that everything is working correctly cut and paste this code into a Jupyter notebook.

```python
import numpy as np
from mongomock import MongoClient
from superduperdb.base.document import Document as D
from superduperdb.components.model import Model
from superduperdb.ext.numpy import array
from superduperdb.backends.mongodb.query import Collection
import superduperdb as s

db = s.superduper(MongoClient().documents)
collection = Collection('docs')

a = array('float64', shape=(32,))

db.execute(
    collection.insert_many([
        D({'x': a(np.random.randn(32))})
        for _ in range(100)
    ]), encoders = (a,)
)

model = Model(
    identifier='test-model',
    object=lambda x: x + 1,
    encoder=a,
)

model.predict(X='x', db=db, select=collection.find())

print(db.execute(collection.find_one()))
```

If this doesn't work then something is wrong 🙉 - please report [an issue on GitHub](https://github.com/SuperDuperDB/superduperdb/issues).