# Datalayer


```{note}
The `Datalayer` is the primary interface from Python to your data+AI.
```

The central object for communicating with MongoDB in SuperDuperDB is the `Datalayer`.

The `Datalayer` object combines 4 basic functionalities involved in integrating AI to the datalayer:

- Querying, updating and inserting data to the datalayer
- Uploading and saving AI models and associated functionality involving large data-blobs
- Saving metadata related to AI models and associated functionality
- (Optional) performing vector search on the datalayer using configured AI models

Correspondingly a datalayer may be built by passing these 4 items to the `Datalayer.__init__` method:

```python
from superduperdb.datalayer.base.datalayer import Datalayer
from superduperdb.datalayer.mongodb.data_backend import MongoDatabackend
from superduperdb.datalayer.mongodb.metadata import MongoMetaDataStore
from superduperdb.datalayer.mongodb.artifacts import MongoArtifactStore
from superduperdb.vector_search.lancedb_client import LanceVectorIndex

import pymongo

mongo_client = pymongo.MongoClient()
my_databackend = MongoDatabackend(mongo_client, name='test_db')
my_metadata = MongoMetaDataStore(mongo_client, name='test_db')
my_artifact_store = MongoArtifactStore(mongo_client, name='_filesystem:test_db')
vector_database = LanceVectorIndex(uri='~/.lancedb')

db = Datalayer(
    data_backend=my_databackend,
    metadata=my_metadata_store,
    artifact_store=my_artifact_store,
    vector_database=my_vector_database,
)
```

Connecting these 4 elements in this way can be slightly tedious, so we provide a helper function to do this on 
the basis of the current configuration (see [here]() for information about configuration).

```python
from superduperdb.datalayer.base.build import build_datalayer

db = build_datalayer()
```

In order to build a datalayer based on MongoDB defaults, a one-size-fits-all wrapper is:

```python
from superduperdb import superduper

db = superduper(pymongo.MongoClient().test_db)
```

## Databackend

Currently we support MongoDB as the data backend.

(artifactstore)=
## Artifact Store

Currently we support `gridfs` on MongoDB as the artifact store.

(metadata)=
## Metadata Store

Currently we support MongoDB as the metadata store.