**`superduperdb.jobs.job`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/jobs/job.py)

## `job` 

```python
job(f)
```
| Parameter | Description |
|-----------|-------------|
| f | function to be decorated |

Decorator to create a job from a function.

## `ComponentJob` 

```python
ComponentJob(self,
     component_identifier: str,
     type_id: str,
     method_name: str,
     args: Optional[Sequence] = None,
     kwargs: Optional[Dict] = None,
     compute_kwargs: Dict = {})
```
| Parameter | Description |
|-----------|-------------|
| component_identifier | unique identifier of the component |
| type_id | type of the component |
| method_name | name of the method to be called |
| args | positional arguments to be passed to the method |
| kwargs | keyword arguments to be passed to the method |
| compute_kwargs | Arguments to use for model predict computation |

Job for running a class method of a component.

## `FunctionJob` 

```python
FunctionJob(self,
     callable: Callable,
     args: Optional[Sequence] = None,
     kwargs: Optional[Dict] = None,
     compute_kwargs: Dict = {})
```
| Parameter | Description |
|-----------|-------------|
| callable | function to be called |
| args | positional arguments to be passed to the function |
| kwargs | keyword arguments to be passed to the function |
| compute_kwargs | Arguments to use for model predict computation |

Job for running a function.

## `Job` 

```python
Job(self,
     args: Optional[Sequence] = None,
     kwargs: Optional[Dict] = None,
     compute_kwargs: Dict = {})
```
| Parameter | Description |
|-----------|-------------|
| args | positional arguments to be passed to the function or method |
| kwargs | keyword arguments to be passed to the function or method |
| compute_kwargs | Arguments to use for model predict computation |

Base class for jobs. Jobs are used to run functions or methods on.

