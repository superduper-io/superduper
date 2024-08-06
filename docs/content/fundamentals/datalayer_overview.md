---
sidebar_position: 3
---

# Datalayer

The `Datalayer` is the principle point of entry in `superduper` for:

- Communicating with the database
- Instructing models and other components to work together with the database
- Accessing and storing meta-data about your `superduper` models and data

Technically, the `Datalayer` "wires together" several important backends involved in the AI workflow:

- Querying the database via the **databackend**
- Storing and retrieving serialized model-weights and other artifacts from the **artifact store**
- Storing and retrieval important meta-data, from the **meta-data store** and information about models and other components which are to be installed with `superduper`
- Performing computations over the data in the **databackend** using the models saved in the **artifact store**

```python
from superduper import superduper

db = superduper()

db.databackend
# <superduper.backends.mongodb.data_backend.MongoDataBackend at 0x1562815d0>

db.artifact_store
# <superduper.backends.mongodb.artifacts.MongoArtifactStore at 0x156869f50>

db.metadata
# <superduper.backends.mongodb.metadata.MongoMetaDataStore at 0x156866a10>

db.compute
# <superduper.backends.local.LocalComputeBackend 0x152866a10>
```

Our aim is to make it easy to set-up each aspect of the `Datalayer` with your preferred
connections/ engines.

### Data-backend

The databackend typically connects to your database (although `superduper` also supports other databackends such as a directory of `pandas` dataframes), 
and dispatches queries written in an query API which is compatible with that databackend, but which also includes additional aspects
specific to `superduper`.

Read more [here](../data_integrations/supported_query_APIs.md).

The databackend is configured by setting the URI `CFG.databackend` in the [configuration system](../get_started/configuration.md).

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


### Compute backend

The compute-backend is designed to be a configurable engine for performing computations with models.
We support 2 backends:

- Local (default: run compute in process on the local machine)
- `dask` (run compute on a configured `dask` cluster)

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
from superduper.backends.local.compute import LocalComputeBackend

db = superduper(
    'sqlite://<my-database>.db',
    metadata_store='sqlite://<my-database>.db',
    artifact_store='filesystem://.superduper/artifacts/',
    compute=LocalComputeBackend(),
)
```

## Key methods

Here are the key methods which you'll use again and again:

### `db.execute`

This method executes a query. For an overview of how this works see [here](../data_integrations/supported_query_APIs.md).

### `db.add`

This method adds `Component` instances to the `db.artifact_store` connection, and registers meta-data
about those instances in the `db.metadata_store`.

In addition, each sub-class of `Component` has certain "set-up" tasks, such as inference, additional configurations, 
or training, and these are scheduled by `db.add`.

<!-- See [here]() for more information about the `Component` class and it's descendants. -->

### `db.show`

This methods displays which `Component` instances are registered with the system.

### `db.remove`

This method removes a `Component` instance from the system.

## Additional methods

### `db.validate`

Validate your components (mostly models)

### `db.predict`

Infer predictions from models hosted by `superduper`. Read more about this and about models [here](../apply_api/model.md).
