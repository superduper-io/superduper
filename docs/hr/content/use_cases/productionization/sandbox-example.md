# SuperDuperDB: cluster usage

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


To build the `sandbox` image:

```bash
make testenv_image
```

> If you want to install additional `pip` dependencies in the image, you have to list them in `requirements.txt`.
> 
> The listed dependencies may refer to:
> 1. standalone packages (e.g `tensorflow>=2.15.0`)
> 2. dependency groups listed in `pyproject.toml` (e.g `.[dev]`)

Then start the environment with:

```bash
make testenv_init
```

This last command starts containers for each of the above services with `docker-compose`. You should see a bunch of logs for each service (mainly MongoDB).

Once you have carried out these steps, you are ready to complete the rest of this notebook, which focuses on a implementing
a production style implementation of vector-search.


```python
import os

# move to the root of the project (assumes starts in `/examples`)
os.chdir('../')

from superduperdb import CFG

# check that config has been properly set-up
assert CFG.data_backend == 'mongodb://superduper:superduper@mongodb:27017/test_db'
```

We'll be using MongoDB to store the vectors and data:


```python
from superduperdb.backends.mongodb import Collection
from superduperdb import superduper

db = superduper()
doc_collection = Collection('documents')
```

We've already prepared some data which was scraped from the `pymongo` query API. You can download it 
in the next cell:


```python
!curl -O https://superduperdb-public.s3.eu-west-1.amazonaws.com/pymongo.json

import json

with open('pymongo.json') as f:
    data = json.load(f)

data[0]
```

Let's insert this data:


```python
from superduperdb import Document

out, G = db.execute(
    doc_collection.insert_many([Document(r) for r in data[:-100]])
)
```

We'll use a `sentence-transformers` model to calculate the embeddings. Here's how to wrap the model 
so that it works with `superduperdb`:


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

Now let's create the vector-search component:


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

This command creates a job on `dask` to calculate the vectors and save them in the database. You can 
follow the `stdout` of this job with this command:


```python
jobs[0].watch()
```

After a few moments, you'll be able to verify that the vectors have been saved in the documents:


```python
db.execute(doc_collection.find_one())
```

Let's test a similarity/ vector search using the hybrid query-API of `superduperdb`. This search 
dispatches one part off to the vector-search server (running on port 8001) and the other (classical) part to MongoDB
the results are combined by `superduperdb`:


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

One of the great things about this distributed setup, is that now allows data to be inserted into the service via other 
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

We can now check that all of the outputs (including those inserted via the `pymongo` client) have been populated 
by the system.


```python
db.execute(doc_collection.count_documents({'_outputs': {'$exists': 1}}))
```
