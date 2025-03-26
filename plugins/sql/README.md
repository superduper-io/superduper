<!-- Auto-generated content start -->
# superduper_sql

superduper-sql is a plugin for SQL databackends that allows you to use these backends with superduper.


Superduper supports SQL databases via the ibis project. With superduper, queries may be built which conform to the ibis API, with additional support for complex data-types and vector-searches.


## Installation

```bash
pip install superduper_sql
```

## API


- [Code](https://github.com/superduper-io/superduper/tree/main/plugins/ibis)
- [API-docs](/docs/api/plugins/superduper_ibis)

| Class | Description |
|---|---|
| `superduper_sql.data_backend.SQLDataBackend` | sql data backend for the database. |


<!-- Auto-generated content end -->

<!-- Add your additional content below -->

## Connection examples

### MySQL

```python
from superduper import superduper

db = superduper('mysql://<mysql-uri>')
```

### Postgres

```python
from superduper import superduper

db = superduper('postgres://<postgres-uri>')
```

### Other databases

```python

from superduper import superduper

db = superduper('<database-uri>')
```