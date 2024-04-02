---
sidebar_position: 4
tags:
  - quickstart
---

# Environment setup

SuperDuperDB may be run in 2 modes:

- **Development**: All functions and computations run by blocking the main thread in the foreground.
- **Cluster**: Long Computations, requests, models run asychronously on separate workers, services

## Development mode

By default, `SuperDuperDB` runs in development mode. This makes it super easy for developers to 
test the code-snippets and use-cases.

In development mode, all computations and configurations take place in a single process. Computations
block the process in the foreground, and developers can easily set breakpoints during computation 
for debugging purposes.

In this mode, connecting to `superduperdb` is as simple as this:

```python
from superduperdb import superduper

db = superduper('<your-database-uri>')
```

## Cluster mode

In cluster mode, the above snippet will not work, since `superduperdb` doesn't currently propagate
this configuration to the rest of the cluster. For that reason, the `data_backend` URI should be specified 
in a [configuration file](configuration.md) common to all services in the cluster, and developers should
connect with:

```python
from superduperdb import superduper

db = superduper()
```

### Services

In cluster mode, multiple individual services are set up which are responsible for various
parts of the work flow:

#### Ray cluster

By specifying a `ray` cluster, computations requested in SuperDuperDB are pushed down to the configured
`ray` cluster, which may be set up with optimized hardware, specific settings, etc.. Read more [here](../cluster_mode/non_blocking_ray_jobs.md)

#### Vector-search service

By specifying a vector-search service, the vector-comparison computation in vector-search queries
is sent to this service, which may be set up to optimize for recall speed and performance.
Read more [here](../cluster_mode/vector_comparison_service.md)

#### Change-data capture service

By specifying a change-data capture service, developers are enabled to 
insert data to their `data_backend` without directly using the `superduperdb` 
package, or even using Python. Read more [here](../cluster_mode/change_data_capture.md).

#### Rest API

By specifying a Rest API service, developers may access `superduperdb` using FastAPI REST 
endpoints, which a documentation and experimentation interface, as well as the 
ability to integrate from non-Python programs. Read more [here](../cluster_mode/rest_service.md).

### Configuration

The configuration file should include the URIs of the services required:

```yaml
# Settings pertaining to cluster mode
cluster:

  # change data capture
  cdc:
    strategy: null
    
    # How to connect to the service
    uri: http://<cdc-host>:<cdc-port>

  # ray compute settings
  compute:

    # How to connect to a ray service
    uri: ray://<ray-host>:<ray-port>

  # vector-search settings
  vector_search:

    # How to connect to the service
    uri: http://<vector_search-host>:<vecto_search-port>
    backfill_batch_size: 100

  # REST API settings (experimental)
  rest:

    # How to connect to the service
    uri: http://<rest-host>:<rest-port>
```

As well as the required database in `data_backend`:

```yaml
data_backend: <database-uri>
```