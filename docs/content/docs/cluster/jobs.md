# Dask Jobs

## Scheduling of training and model outputs

In order to most efficiently marshall computational resources,
SuperDuperDB may be configured to run in asynchronous mode `{"distributed": True}`.
The simplesst way to set a distributed SuperDuperDB deployment is using a [single-host cluster](single_host_cluster). See [the section on configuration](configuration) for details in setting up SuperDuperDB.

There are several key functionalities in SuperDuperDB which trigger asynchronous jobs to be
spawned in the configured Dask worker pool.

- Inserting data
- Updating data
- Creating `Listener` instances
- Apply models to data `model.predict`
- Training models `model.fit`

See [the Dask documentation](https://docs.dask.org/en/stable/) for more information about setting up and managing Dask deployments. The dask deployment may be configured using 
the [configuration stystem](configuration).

The stdout and status of the job may be monitored using the returned `Job` object:

```python
>>> job = model.predict(X='my-key', db=db, select=collection.find())
>>> job.listen()
# ... lots of lines of stdout
```

Jobs may be viewed using `db.show`:

```python
>>> db.show('job')
```