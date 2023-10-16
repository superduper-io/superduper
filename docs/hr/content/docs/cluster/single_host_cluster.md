# Single host cluster deployment

The simplest way to create a SuperDuperDB deployment, is to use the CLI 
to start all of the services involved in one command:

```bash
python -m superduperdb local_cluster [collection_names] 
```

This command starts the following components:

- Server (accessible via the SuperDuperDB client)
- Local Dask Cluster (configurable via configuration system) for deploying jobs
- Change data capture implementing [listeners](/docs/docs/usage/models#daemonizing-models-with-listeners) on MongoDB collections mentioned in space separated list `collection_names`

See our architecture diagram [here](architecture) for a more detailed explanation of how 
these components interact.