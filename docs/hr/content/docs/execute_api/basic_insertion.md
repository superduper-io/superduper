# Basic insertion

SuperDuperDB supports inserting data wrapped as dictionaries in Python.
These dictionaries may contain basic JSON-compatible data, but also 
other data-types to be handled with `DataType` components. All data inserts are wrapped with the `Document` wrapper:

```python
from superduperdb import Document
data = ... # an iterable of dictionaries
data = [Document(r) for r in data]
```

## MongoDB

```python
ids, jobs = db.execute(
    my_collection.insert_many(data)
)
```

## SQL

```python
ids, jobs = db.execute(
    my_table.insert(data)
)
```

## Monitoring jobs

The second output of this command gives a reference to the job computations 
which are triggered by the `Component` instances already applied with `db.apply(...)`.

If users have configured a `ray` cluster, the jobs may be monitored at the 
following uri:

```python
from superduperdb import CFG

print(CFG.cluster.compute.uri)
```