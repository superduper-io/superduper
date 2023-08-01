# When to use standalone or cluster mode

SuperDuperDB may be used in **cluster** or **standalone** mode, depending on the use-case, and whether SuperDuperDB is being used for development of production. By default SuperDuperDB runs in **standalone** mode.
**Cluster** and **standalone** mode affect how and where computations are run, which are triggered whenever:

- Data is inserted or updated, triggering data to be downloaded and `model.predict` to be executed on added `Listener` instances.
- `model.fit` is called

By using a combination of **standalone** mode and **cluster** mode, users and teams may profit from the advantages of both aspects, especially when transitioning from development to deployment.

## Standalone

In standalone mode, all computations and queries run in a single thread. 

***Advantages***

- Computations are run transparently in the foreground
- Debugging is straightforward
- Works well in local deployments on a single client

***Disadvantages***

- Solution does not scale to arbitrary numbers of workers
- No solution for communication with remote hosts/ apps
- No insertion of data except through SuperDuperDB main thread

## Cluster

In cluster mode, all computations run asynchronously on a [Dask cluster](jobs). In addition, 
a [change-data capture](CDC) service (CDC) is run on a separate thread, which listens to MongoDB for changes.
Read more [here](clustersection).

***Advantages***

- Compute is scalable to magnitude handled by Dask cluster
- Deployment is accessible remotely via client-server implementation

***Disadvantages***

- Debugging is challenging, since breakpoints cannot be set easily in workers
- Managing solution requires setting up logging service to monitor all components


