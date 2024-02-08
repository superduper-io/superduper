---
sidebar_position: 3
---

# Running non-blocking computations in the background

`superduperdb` offers the possiblity to run all long running blocking jobs in the background via `ray`.
Read about the `ray` project [here](https://www.dask.org/).

To configure this feature, configure:

```python
from superduperdb import CFG

CFG.mode = 'production'
```

When this is so-configured the following functions push their computations to the `ray` cluster:

- `db.add`
- `db.insert`
- `db.update`
- `Model.predict`
- `Model.fit`

When `ray` is configured, these functions returns either a `superduperdb.job.Job` object, or an iterable thereof.

```python
job = m.predict(     # a `superduper.job.ComponentJob` object
    X='x',
    db=db,
    select=Collection('localcluster').find(),
)

job.watch()          # watch the `stdout` of the `Job`
```