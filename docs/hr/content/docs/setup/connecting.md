---
sidebar_position: 2
tags:
  - quickstart
---

# Connecting

:::info
In this document we instantiate the variable `db` based on configuration and overrides.
In the remainder of the documentation, we reuse this variable without comment
:::

The simplest way to connect to `superduperdb` is with:

```python
from superduperdb import superduper
db = superduper()
```

This command uses settings inherited from [the configurations set previously](./configuration.md).
In order to connect to a different database, one can do:

```python
db = superduper('mongodb://localhost:27018')
```

Additional configurations can be injected with `**kwargs`

```python
db = superduper('mongodb://localhost:27018', artifact_store='filesystem://./data')
```

... or by passing a modified `CFG` object.

```python
from superduperdb import CFG

CFG.artifact_store = 'filesystem://./data'
db = superduper('mongodb://localhost:27018', CFG=CFG)
```

The `db` object is an instance of `superduperdb.base.datalayer.Datalayer`.
The `Datalayer` class handles AI models and communicates with the databackend and associated components. Read more [here](../fundamentals/datalayer_overview.md).
