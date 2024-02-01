# DuckDB

`superduperdb` supports DuckDB via the [`ibis` project](https://ibis-project.org/).
With `superduperdb`, queries may be built which conform to the `ibis` API, with additional 
support for complex data-types and vector-searches.

## Setup

The first step in working with DuckDB, is to define a table and schema.

```python
from superduperdb.backends.ibis.query import Table
from superduperdb.backends.ibis.field_types import dtype
from superduperdb import Schema

db = superduper('duckdb://test.dbb')

t = Table(
    'my_table',
    primary_id='id',
    schema=Schema(
        'table-schema',
        fields={'id': dtype(str),
                'custom_text': dtype(str),
                'custom_integer': dtype(int)},
    )
)

db.add(t)
```

## Inserting data

Table data must correspond to the `Schema` for that table:

```python
import pandas

df = pandas.DataFrame([
       [1, "some-custom-text", 3],
       [2, "moar-custom-test", 2],
       [3, "guess what? this is 'custom text'", 4],
       [4, "I can't believe this isn't custom text", 8,]
    ], columns=["id", "custom_text", "custom_integer"])

db.execute(t.insert(df))
```

## Selecting data

`superduperdb` supports selecting data via the `ibis` query API.

The following are equivalent:

```python
results = db.execute(t.filter(t.custom_integer > 3).limit(5))
results.as_pandas()["custom_text"].values.tolist()
```

### Vector-search

Vector-searches are supported via the `like` operator:

```python
db.execute(
    t.like({'custom_text': 'moar'}, vector_index='my-index')
     .limit(5)
     .select(t.custom_integer, t.id)
)
```

Vector-searches are either first or last in a chain of operations:

```python
db.execute(
    t.filter(t.custom_integer > 3)
     .limit(5)
     .select(t.custom_text, t.id)
     .like({'custom_text': 'guess what?'}, vector_index='my-index')
)
```

### Support for raw-sql

... the first query above is equivalent to:

```python
from superduperdb.backends.ibis.query import RawSQL

db.execute(RawSQL('SELECT custom_text FROM my_table WHERE custom_integer > 3 LIMIT 5;'))
```

... the second will be equivalent to:

```python
from superduperdb.backends.ibis.query import RawSQL

raw_sql = RawSQL(
    '''
    SELECT custom_integer FROM my_table 
    LIKE text = 'moar'
    LIMIT 5;
    '''
    )

db.execute(raw_sql)
```

## Updating data

Updates are not covered for `superduperdb` SQL integrations.

## Deleting data

```python
db.databackend.drop_table('my_table')
```