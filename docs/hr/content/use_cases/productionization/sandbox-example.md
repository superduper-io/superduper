# Cluster usage

SuperDuperDB allows developers, on the one hand to experiment and setup models quickly in scripts and notebooks, and on the other hand deploy persistent services, which are intended to "always" be on. These persistent services are:

- Dask scheduler
- Dask workers
- Vector-searcher service
- Change-data-capture (CDC) service

![](/img/light.png)

To set up `superduperdb` to use this cluster mode, it's necessary to add explicit configurations 
for each of these components. The following configuration does that, as well as enabling a pre-configured 
community edition MongoDB database:

```yaml
data_backend: mongodb://superduper:superduper@mongodb:27017/test_db
artifact_store: filesystem://./data
cluster:
  cdc: http://cdc:8001
  compute: dask+tcp://scheduler:8786
  vector_search: http://vector-search:8000
```

Add this configuration in `/.superduperdb/config.yaml`, where `/` is the root of your project.

Once this configuration has been added, you're ready to use the `superduperdb` sandbox environment, which uses 
`docker-compose` to deploy:

- Standalone replica-set of MongoDB community edition
- Dask scheduler
- Dask workers
- Vector-searcher service
- Change-data-capture (CDC) service
- Jupyter notebook service

To set up this environment, navigate to your local copy of the `superduperdb` repository, and build the image with:

```bash
make testenv_image SUPERDUPERDB_EXTRAS=sandbox
```

Then start the environment with:

```bash
make testenv_init
```

This last command starts containers for each of the above services with `docker-compose`. You should see a bunch of logs for each service (mainly MongoDB).

Once you have carried out these steps, you are ready to complete the rest of this notebook.


```python
import os

# move to the root of the project (assumes starts in `/examples`)
os.chdir('../')

from superduperdb import CFG

# check that config has been properly set-up
assert CFG.data_backend == 'mongodb://superduper:superduper@mongodb:27017/test_db'
```


```python
from superduperdb.backends.mongodb import Collection
from superduperdb import superduper

db = superduper()
doc_collection = Collection('documents')
```


```python
!curl -O https://superduperdb-public.s3.eu-west-1.amazonaws.com/pymongo.json

import json

with open('pymongo.json') as f:
    data = json.load(f)

data[0]
```


```python
from superduperdb import Document

out, G = db.execute(
    doc_collection.insert_many([Document(r) for r in data[:-100]])
)
```


```python
db.metadata.show_jobs()
```


```python
import sentence_transformers
from superduperdb import Model, vector

model = Model(
   identifier='all-MiniLM-L6-v2',
   object=sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2'),
   encoder=vector(shape=(384,)),
   predict_method='encode',           # Specify the prediction method
   postprocess=lambda x: x.tolist(),  # Define postprocessing function
   batch_predict=True,                # Generate predictions for a set of observations all at once 
)
```


```python
from superduperdb import Listener, VectorIndex

jobs, vi = db.add(
    VectorIndex(
        identifier=f'pymongo-docs-{model.identifier}',
        indexing_listener=Listener(
            select=doc_collection.find(),
            key='value',
            model=model,
            predict_kwargs={'max_chunk_size': 1000},
        ),
    )
)
```


```python
jobs[0].watch()
```


```python
db.execute(doc_collection.find_one())
```


```python
from IPython.display import Markdown

result = sorted(db.execute(
    doc_collection
        .like(Document({'value': 'Aggregate'}), n=10, vector_index=f'pymongo-docs-{model.identifier}')
        .find({}, {'_outputs': 0})
), key=lambda r: -r['score'])

# Display a horizontal line
display(Markdown('---'))

# Iterate through the query results and display them
for r in result:
    # Display the document's parent and res values in a formatted way
    display(Markdown(f'### `{r["parent"] + "." if r["parent"] else ""}{r["res"]}`'))
    
    # Display the value of the document
    display(Markdown(r['value']))
    
    # Display a horizontal line
    display(Markdown('---'))
```


```python
db.drop(force=True)
```

The great thing about this production mode, is that now allows data to be inserted into the service via other 
MongoDB clients, even from other programming languages and applications.

We show-case this here, by inserting the rest of the data using the official Python MongoDB driver `pymongo`.

This cell will update the models, even if you restart the program:


```python
import pymongo

coll = pymongo.MongoClient('mongodb://superduper:superduper@mongodb:27017/test_db').test_db.documents

coll.insert_many(data[-100:])
```

To get an idea what is happening, you can view the logs of the CDC container on 
your host by executing this command in a terminal:

```bash
docker logs -n 20 testenv_cdc_1
```

Note this won't work inside this notebook since it's running in its own container.

The CDC service should have captured the changes created with the `pymongo` insert, and has submitted a new job(s)
to the `dask` cluster.

You can confirm that another job has been created and executed:


```python
db.metadata.show_jobs()
```

You can view the `stdout` of the most recent job with this command:


```python
db.metadata.watch_job('a5077d81-0e00-4004-b501-23af356e0234')
```


```python
db.execute(doc_collection.count_documents({'_outputs': {'$exists': 1}}))
```
