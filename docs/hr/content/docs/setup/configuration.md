---
sidebar_position: 1
tags:
  - quickstart
---

# Configure

SuperDuperDB provides a range of configurable options for setting
up your environment:

Configurations can either be injected:

- in a YAML file: `.superduperdb/config.yaml` or
- through environment variables starting with `SUPERDUPERDB_`:
- as `**kwargs` when calling the [`superduperdb.superduper`](./connecting.md) function.

Here are the configurable settings and their project defaults 
(remaining configurations can be viewed in [`superduperdb.base.config`](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/base/config.py)). Note that as much or as little of this configuration can be specified. The remaining 
configurations will then take on their default values.


```yaml

# URI of your database/ datastore (here the name of the db is `test_db`)
data_backend: mongodb://superduper:superduper@localhost:27017/test_db

# URI of your artifact store (defaults to `data_backend`)
artifact_store: null

# URI of your metadata store (defaults to `data_backend`)
metadata_store: null

# options for setting up a distributed cluster setup
cluster:
  
  # Size of chunks syncing data-base with vector-search
  backfill_batch_size: 100

  # URI of CDC service (default - no CDC)
  cdc: null
  
  # URI of compute resource (defaults to in-process)
  compute: local://

  # URI of vector-search service (defaults to numpy in-process)
  vector_search: in_memory://

  # Location of local search indices for lance
  lance_home: .superduperdb/vector_indices

# Location of hybrid data (if `null` then no hybrid storage)
downloads_folder: null

# Probability of new data assigned to "valid"
fold_probability: 0.05

# Log-level DEBUG/ INFO
log_level: DEBUG

# Logging-type
logging_type: SYSTEM

# Parameters for retrying connections
retries:
  stop_after_attempt: 2
  wait_max: 10.0
  wait_min: 4.0
  wait_multiplier: 1.0
```

As an example, to reconfigure the URI of the data_backend we have two options:

A configuration file `.superduperdb/config.yaml` with this content only:

```yaml
data_backend: mongodb://localhost:27018/documents
```

... or

```bash
$ export SUPERDUPERDB_DATA_BACKEND='mongodb://localhost:27018/documents'
```

You may view the configuration used by the system with:

```bash
python -m superduperdb config
```