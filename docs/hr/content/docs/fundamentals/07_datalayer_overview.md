---
sidebar_position: 7
---

# Datalayer

The abstraction coordinates models, data and backends is the `superduperdb.base.datalayer.Datalayer` class.

The `Datalayer` is a highly configurable class which "wires together" three important backends involved in the AI workflow:

- Querying the database via the **databackend**
- Storing and retrieving serialized model-weights and other artifacts from the **artifact store**
- Storing and retrieval important meta-data, from the **meta-data store** and information about models and other components which are to be installed with `superduperdb`

```python
from superduperdb import superduper

db = superduper()

db.databackend
# <superduperdb.backends.mongodb.data_backend.MongoDataBackend at 0x1562815d0>

db.artifact_store
# <superduperdb.backends.mongodb.artifacts.MongoArtifactStore at 0x156869f50>

db.metadata
# <superduperdb.backends.mongodb.metadata.MongoMetaDataStore at 0x156866a10>
```

Our aim is to make it easy to set-up each aspect of the `Datalayer` with your preferred
databases.

### Data-backend

The databackend typically connects to your database (although `superduperdb` also supports other databackends such as a directory of `pandas` dataframes), 
and dispatches queries written in an query API which is compatible with that databackend, but which also includes additional aspects
specific to `superduperdb`.

Read more [here](../walkthrough/11_supported_query_APIs.md).

The databackend is configured by setting the URI `CFG.databackend` in the [configuration system](../walkthrough/01_configuration.md).

We support the same databackends as supported by the [`ibis` project](https://ibis-project.org/):

- [**MongoDB**](https://www.mongodb.com/)
- [**PostgreSQL**](https://www.postgresql.org/)
- [**SQLite**](https://www.sqlite.org/index.html)
- [**DuckDB**](https://duckdb.org/)
- [**BigQuery**](https://cloud.google.com/bigquery)
- [**ClickHouse**](https://clickhouse.com/)
- [**DataFusion**](https://arrow.apache.org/datafusion/)
- [**Druid**](https://druid.apache.org/)
- [**Impala**](https://impala.apache.org/)
- [**MSSQL**](https://www.microsoft.com/en-us/sql-server/)
- [**MySQL**](https://www.mysql.com/)
- [**Oracle**](https://www.oracle.com/database/)
- [**pandas**](https://pandas.pydata.org/)
- [**Polars**](https://www.pola.rs/)
- [**PySpark**](https://spark.apache.org/docs/3.3.1/api/python/index.html)
- [**Snowflake**](https://www.snowflake.com/en/)
- [**Trino**](https://trino.io/)

### Artifact Store

The artifact-store is the place where large pieces of data associated with your AI models are saved.
Users have the possibility to configure either a local filesystem, or an artifact store on MongoDB `gridfs`:

For example:

```python
CFG.artifact_store = 'mongodb://localhost:27017/documents'
```

Or:

```python
CFG.artifact_store = 'filesystem://./data'
```

### Metadata Store

The meta-data store is the place where important information associated with models and 
related components are kept:

- Where are the data artifacts saved for a component?
- Important parameters necessary for using a component
- Important parameters which were used to create a component (e.g. in training or otherwise)

Similarly to the databackend and artifact store, the metadata store is configurable:

```python
CFG.metadata = 'mongodb://localhost:27017/documents'
```

We support metadata store via:

1. [MongoDB](https://www.mongodb.com/)
1. All databases supported by [SQLAlchemy](https://www.sqlalchemy.org/).
   For example, these databases supported by the databackend are also supported by the metadata store.
   - [PostgreSQL](https://www.postgresql.org/)
   - [MySQL](https://www.mysql.com/)
   - [SQLite](https://www.sqlite.org/)
   - [MSSQL](https://www.microsoft.com/en-us/sql-server/sql-server-downloads)


## Default settings

In such cases, the default configuration is to use the same configuration as used in the 
databackend.

I.e., for MongoDB the following are equivalent:

```python
db = superduper('mongodb://localhost:27018/documents')
```

...and

```python
db = superduper(
    'mongodb://localhost:27018/documents',
    metadata_store='mongodb://localhost:27018/documents',
    artifact_store='mongodb://localhost:27018/documents',
)
```

Whenever a database is supported by the artifact store and metadata store, 
the same behaviour holds. However, since there is no general pattern
for storing large files in SQL databases, the fallback artifact store
is on the local filesystem. So the following are equivalent:

```python
db = superduper('sqlite://<my-database>.db')
```

...and

```python
db = superduper(
    'sqlite://<my-database>.db',
    metadata_store='sqlite://<my-database>.db',
    artifact_store='filesystem://.superduperdb/artifacts/',
)
```
