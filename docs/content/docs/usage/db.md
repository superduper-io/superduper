---
sidebar_position: 1
---

# DB

```{note}
The `DB` is the primary interface from Python to your data+AI.
```

The central object for communicating with your datastore in SuperDuperDB is the `DB` class.

The `DB` object combines 4 basic functionalities involved in integrating AI to the database:

- Querying, updating and inserting data 
- Uploading and saving AI models and associated functionality involving large data-blobs
- Saving metadata related to AI models and associated functionality
- (Optional) performing vector search on the database using configured AI models

Correspondingly a database may be built by passing these 4 items to the `DB.__init__` method:

```python
from superduperdb.db.base.db import DB
from superduperdb.db.mongodb.data_backend import MongoDatabackend
from superduperdb.db.mongodb.metadata import MongoMetaDataStore
from superduperdb.db.mongodb.artifacts import MongoArtifactStore
from superduperdb.vector_search.lancedb_client import LanceVectorIndex

import pymongo

mongo_client = pymongo.MongoClient()
my_databackend = MongoDatabackend(mongo_client, name='test_db')
my_metadata = MongoMetaDataStore(mongo_client, name='test_db')
my_artifact_store = MongoArtifactStore(mongo_client, name='_filesystem:test_db')
vector_database = LanceVectorIndex(uri='~/.lancedb')

db = DB(
    data_backend=my_databackend,
    metadata=my_metadata_store,
    artifact_store=my_artifact_store,
    vector_database=my_vector_database,
)
```

Connecting these 4 elements in this way can be slightly tedious, so we provide a helper function to do this on 
the basis of the current configuration.

```python
from superduperdb.db.base.build import build_datalayer

db = build_datalayer()
```

In order to build a database based on defaults, a one-size-fits-all wrapper is:

```python
from superduperdb import superduper

db = superduper(pymongo.MongoClient().test_db)
```

## Databackend

Currently we support MongoDB as the data backend, with large data blobs on AWS s3.

(artifactstore)=
## Artifact Store

Currently we support `gridfs` on MongoDB as the artifact store.

(metadata)=
## Metadata Store

Currently we support MongoDB as the metadata store.