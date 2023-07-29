(clientserver)=
# Client-server implementation

So that users are able to access SuperDuperDB from remote servers/ client-side,
we provide a client-server implementation to do this. Note that this is not
strictly necessary to profit from SuperDuperDB functionality. Another
logical usage pattern for remote servers, would be to access the environment
from outside using a Jupyter notebook service, deployed with local
network access to the Dask cluster and change data capture components.

To start the server, use the CLI:

```bash
python -m superduperdb server
```

The client may be used from a Python program as follows:

```python
from superduperdb.cluster.client import Client
from superduperdb import superduper
from superduperdb.datalayer.mongodb.query import Collection

c = Client(uri='<uri>')
collection = Collection(name='docs')

c.show('model')           # standard methods supported by `DataLayer` are accessible here
r = c.execute(collection.find_one())       # data fetched with standard queries
```
