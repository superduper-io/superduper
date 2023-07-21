# Minimum working example

To check that everything is working correctly try the notebook "minimum-working-example.ipynb"
in the `notebooks/` directory. For completeness, here is the code to execute:

```python
import numpy as np
from pymongo import MongoClient
from superduperdb.core.documents import Document as D
from superduperdb.encoders.numpy.array import array
from superduperdb.datalayer.mongodb.query import Collection, InsertMany
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

print(db.execute(collection.find_one()))
```
