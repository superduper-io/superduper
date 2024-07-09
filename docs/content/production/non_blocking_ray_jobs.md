---
sidebar_position: 3
---

# Running non-blocking Ray computations in the background

`superduper` offers the possiblity to run all long running blocking jobs in the background via `ray`.
Read about the `ray` project [here](https://www.ray.io/).

To configure this feature, configure:

```yaml
cluster:
  compute:
    uri: ray://<ray_host>:<ray_port>
```

When this is so-configured the following functions push their computations to the `ray` cluster:

- `db.apply`
- `db.execute` (if data is inserted, deleted, updated)

When `ray` is configured, these functions returns either a `superduper.job.Job` object, or an iterable thereof.

```python
jobs = db.apply(<component>)[0]
```