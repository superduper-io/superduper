# Run computations on Dask

In this example, we show how to run computations on a Dask cluster, rather than in the same process as 
data is submitted from. This allows compute to be scaled horizontally, and also submitted to 
workers, which may utilize specialized hardware, including GPUs.

To do this, we need to override the default configuration. To do this, we only need specify the 
configurations which diverge from the defaults. In particular, to use a Dask cluster, we specify 
`CFG.distributed = True`


```python
!echo '{"distributed": true}' > configs.json
!cat configs.json
```

We can now confirm, by importing the loaded configuration `CFG`, that `CFG.distribute == True`:


```python
from superduperdb import CFG

import pprint
pprint.pprint(CFG.dict())
```

Now that we've set up the environment to use a Dask cluster, we can add some data to the `Datalayer`.


```python
from superduperdb.db.base.build import build_datalayer

db = build_datalayer()
```


```python
db.db.client.drop_database('test_db')
db.db.client.drop_database('_filesystem:test_db')
```

As in the previous tutorials, we can wrap models from a range of AI frameworks to interoperate with the data set, 
as well as inserting data with, for instances, tensors of a specific data type:


```python
import pymongo
import torch

from superduperdb import superduper
from superduperdb.container.document import Document as D
from superduperdb.ext.torch.tensor import tensor
from superduperdb.db.mongodb.query import Collection

m = superduper(
    torch.nn.Linear(128, 7),
    encoder=tensor(torch.float, shape=(7,))
)

t32 = tensor(torch.float, shape=(128,))

output = db.execute(
    Collection('localcluster').insert_many(
        [D({'x': t32(torch.randn(128))}) for _ in range(1000)], 
        encoders=(t32,)
    )
)
```

Now when we instruct the model to make predictions based on the `Datalayer`, the computations run on the Dask cluster. The `.predict` method returns a `Job` instance, which can be used to monitor the progress of the computation:


```python
job = m.predict(
    X='x',
    db=db,
    select=Collection('localcluster').find(),
)

job.watch()
```

To check that the `Datalayer` has been populated with outputs, we can check the `"_outputs"` field of a record:


```python
db.execute(Collection('localcluster').find_one()).unpack()
```
