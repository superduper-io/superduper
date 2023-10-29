---
sidebar_position: 1
---

# DB

:::info
The `DB` is the primary interface from Python to your data+AI.
:::

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

class SuperDuperDatabase:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SuperDuperDatabase, cls).__new__(cls)
            # Create necessary objects to build the database
            mongo_client = pymongo.MongoClient()
            data_backend = MongoDatabackend(mongo_client, name='test_db')
            metadata_store = MongoMetaDataStore(mongo_client, name='test_db')
            artifact_store = MongoArtifactStore(mongo_client, name='_filesystem:test_db')
            vector_database = LanceVectorIndex(uri='~/.lancedb')
            # Initialize the DB class
            cls._instance.db = DB(
                data_backend=data_backend,
                metadata=metadata_store,
                artifact_store=artifact_store,
                vector_database=vector_database
            )
        return cls._instance
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

Currently we support:

- [MongoDB](https://www.mongodb.com/)
- (Experimental) Any databackend supported by the [Ibis project](https://ibis-project.org/), including:
  - [SQLite](https://www.sqlite.org/index.html)
  - [PostgreSQL](https://www.postgresql.org/)
  - [Snowflake](https://www.snowflake.com/en/)
  - [DuckDB](https://duckdb.org/)
  - [Pandas](https://pandas.pydata.org/)
  - ... many more

## Artifact Store

Currently we support:
- MongoDB via `gridfs`
- Local filesystem

## Metadata Store

- MongoDB
- Any SQL database supported by SQLAlchemy
