**`superduperdb.jobs.tasks`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/jobs/tasks.py)

## `callable_job` 

```python
callable_job(cfg,
     function_to_call,
     args,
     kwargs,
     job_id,
     dependencies=(),
     db: Optional[ForwardRef('Datalayer')] = None)
```
| Parameter | Description |
|-----------|-------------|
| cfg | configuration |
| function_to_call | function to call |
| args | positional arguments to pass to the function |
| kwargs | keyword arguments to pass to the function |
| job_id | unique identifier for this job |
| dependencies | other jobs that this job depends on |
| db | datalayer to use |

Run a function in the database.

## `method_job` 

```python
method_job(cfg,
     type_id,
     identifier,
     method_name,
     args,
     kwargs,
     job_id,
     dependencies=(),
     db: Optional[ForwardRef('Datalayer')] = None)
```
| Parameter | Description |
|-----------|-------------|
| cfg | user config |
| type_id | type of component |
| identifier | identifier of component |
| method_name | name of method to run |
| args | positional arguments to pass to the method |
| kwargs | keyword arguments to pass to the method |
| job_id | unique identifier for this job |
| dependencies | other jobs that this job depends on |
| db | datalayer to use |

Run a method on a component in the database.

