# SuperDuperDB Cluster

SuperDuperDB may be operated as a cluter in `distributed` mode, with a client-server
for communication for production use-cases.
In this section of the documentation we describe how to set this up.

In cluster mode, several features become available to developers, 
leading to smoother and more robust productionization:

- Task parallelization using `dask`
- Change data capture using MongoDB [change streams](https://www.mongodb.com/docs/manual/changeStreams/).
- A client-server implementation, allowing remote access to a SuperDuperDB deployment.

## Contents

```{toctree}
:maxdepth: 2

configuration
single_host_cluster
architecture
jobs
change_data_capture
client_server
distributed_cluster
```
