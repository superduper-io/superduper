---
sidebar_position: 3
tags:
  - quickstart
---

# Configure

SuperDuperDB provides a range of configurable options for setting
up your environment:

Configurations can either be injected:

- in a YAML file specified by the `SUPERDUPERDB_CONFIG_FILE` environment variable or
- through environment variables starting with `SUPERDUPERDB_`:
- as `**kwargs` when calling the [`superduperdb.superduper`](./connecting.md) function (note this is only for development purposes).

Here are the configurable settings and their project defaults 
(remaining configurations can be viewed in [`superduperdb.base.config`](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/base/config.py)). Note that as much or as little of this configuration can be specified. The remaining 
configurations will then take on their default values.


```yaml
# Where large data blobs/ files are saved
artifact_store: filesystem://<path-to-artifact-store>

# How to encode binary data
bytes_encoding: Bytes
# bytes_encoding: Base64

# Settings pertaining to cluster mode
cluster:

  # change data capture
  cdc:
    strategy: null
    
    # (optional) How to connect to the service
    uri: None
    # uri: http://<host>:<port>

  # ray compute settings
  compute:

    # (optional) How to connect to a ray service
    uri: None
    # uri: ray://<host>:<port>

  # vector-search settings
  vector_search:

    # (optional) How to connect to the service
    uri: None
    # uri: http://<host>:<port>
    backfill_batch_size: 100

  # (optional) REST API settings (experimental)
  rest:

    # How to connect to the service
    uri: None
    # uri: http://<host>:<port>

    config: None
    # config: path/to/rest_config.yaml

# The base database you would like to connect to
data_backend: <databackend-uri>

# Settings pertaining to downloading data from URIs
downloads:
  folder: null
  headers:
    User-Agent: me
  n_workers: 0
  timeout: null

# Settings for randomly assigning train/valid folds
fold_probability: 0.05

# Where lance indexes will be saved
lance_home: .superduperdb/vector_indices

# Log level to be shown to stdout
log_level: INFO

logging_type: SYSTEM

# Database to save meta-data in (defaults to `data_backend`)
metadata_store: null

# Settings for failed API requests
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