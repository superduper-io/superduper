---
sidebar_position: 3
---

# SQL

`superduperdb` supports SQL databases via the [`ibis` project](https://ibis-project.org/).
With `superduperdb`, queries may be built which conform to the `ibis` API, with additional 
support for complex data-types and vector-searches.

## Setup

The first step in working with an SQL table, is to define a table and schema

```python
from superduperdb.backends.ibis.query import Table
from superduperdb.backends.ibis.field_types import dtype
from superduperdb import Encoder, Schema

my_enc = Encoder('my-enc')

schema = Schema('my-schema', fields={'img': my_enc, 'text': dtype('str'), 'rating': dtype('int')})

db = superduper()

t = Table('my-table', schema=schema)

db.add(t)
```

## Inserting data

Table data must correspond to the `Schema` for that table:

```python
import pandas

pandas.DataFrame([
    PIL.Image.open('image.jpg'), 'some text', 4,
    PIL.Image.open('other_image.jpg'), 'some other text', 3,
])

t.insert(dataframe)
```

## Selecting data

`superduperdb` supports selecting data via the `ibis` query API.

The following are equivalent:

```python
db.execute(
    t.filter(t.rating > 3).limit(5).select(t.image)
)
```

### Vector-search

Vector-searches are supported via the `like` operator:

```python
db.execute(
    t.like({'text': 'something like this'}, vector_index='my-index')
     .filter(t.rating > 3)
     .limit(5)
     .select(t.image, t.id)
)
```

Vector-searches are either first or last in a chain of operations:

```python
db.execute(
    t.filter(t.rating > 3)
     .limit(5)
     .select(t.image, t.id)
     .like({'text': 'something like this'}, vector_index='my-index')
)
```

### Support for raw-sql

... the first query above is equivalent to:

```python
from superduperdb.backends.ibis.query import RawSQL

db.execute(RawSQL('SELECT img FROM my-table WHERE rating > 3 LIMIT 5;'))
```

... the second will be equivalent to:

```python
from superduperdb.backends.ibis.query import RawSQL

raw_sql = RawSQL(
    '''
    SELECT img FROM my-table 
    LIKE text = 'something like this'
    WHERE rating > 3
    LIMIT 5;
    '''
    )

db.execute(raw_sql)
```

## Updating data

Updates are not covered for `superduperdb` SQL integrations.

## Deleting data

```python
db.databackend.drop_table('my-table')
```
