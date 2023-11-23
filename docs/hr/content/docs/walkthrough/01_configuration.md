---
sidebar_position: 1
tags:
  - quickstart
---

# Configure

The first step in SuperDuperDB is to configure a flexible range of options for setting
up your environment:

Configurations can either be injected:

- directly in Python using the `superduperdb.CFG` data class
- in a YAML file: `.superduperdb/config.yaml` or
- through environment variables starting with `SUPERDUPERDB_`:

Here are the configurable settings and their project defaults 
(remaining configurations can be viewed in [`superduperdb.base.config`](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/base/config.py)).

```yaml
# URI of your database
# All URIs are of the form:
# <db_prefix>://<user-and-password-and-host-info>/<database-name>
data_store_uri: mongodb://localhost:27017/documents

# in_memory/ lance/ or CFG.data_store_uri vector-comparison implementation
vector_search: in_memory

# URI of your artifact store (defaults to `data_store_uri`)
artifact_store: null

# URI of your metadata store (defaults to `data_store_uri`)
metadata_store: null

# Probability of new data assigned to "valid"
fold_probability: 0.05
    
# URI of dask_scheduler
dask_scheduler: tcp://localhost:8786

# development or production mode
production: false

# hybrid storage or not (set to a directory to used hybrid storage)
hybrid_data: null
```

For example, to configure a connection to a MongoDB database "documents" on `localhost` at port `27018`, these are equivalent:

In Python

```python
from superduperdb import CFG

CFG.data_store_uri = 'mongodb://localhost:27018/documents
```

... or

```bash
$ export SUPERDUPERDB_DATA_STORE_URI='mongodb://localhost:27018/documents'
$ python -c 'import superduperdb; print(superduperdb.CFG.databackend)'
mongodb://localhost:27018/documents
```