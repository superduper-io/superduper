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

Vector-searches are supported via the `like` operator. Let's prepare a `Model` and a `VectorIndex`first:

```python
from sentence_transformers import SentenceTransformer
from superduperdb.ext.numpy import array
from superduperdb import Listener, VectorIndex, Model

model = Model(
    identifier='all-MiniLM-L6-v2',
    object=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'),
    preprocess=lambda r: r,
    encoder=array(dtype='float32', shape=(384,)),
    predict_method='encode',
    batch_predict=True,
    device='mps')

db.add(
    VectorIndex(
        identifier='my-index',
        indexing_listener=Listener(
            select=t,
            key='custom_text',
            model=model,
            predict_kwargs={'max_chunk_size': 500, 'batch_size': 30},
        ),
    )
)
```

You can perform a vector search:

```python
from superduperdb import Document

res = db.execute(
    t
    .like(Document({'custom_text': 'custom'}), vector_index='my-index', n=10)
    .limit(10)
)
res.as_pandas()
```

### Support for raw-sql

Raw SQL can be used to select data from the database.
Here is an example inline:

```python
from superduperdb.backends.ibis.query import RawSQL

results = db.execute(
            RawSQL('SELECT custom_text FROM my_table WHERE custom_integer > 3 LIMIT 5;')
            )
results.as_pandas()["custom_text"].values.tolist()
```

Here is another example:

```python
from superduperdb.backends.ibis.query import RawSQL

raw_sql = RawSQL(
    '''
    SELECT custom_integer FROM my_table 
    WHERE custom_text LIKE '%moar%'
    LIMIT 5;
    '''
    )

results = db.execute(raw_sql)
results.as_pandas()["custom_integer"].values.tolist()
```

## Updating data

Updates are not covered for `superduperdb` SQL integrations.

## Deleting data

```python
raw_sql = RawSQL(
    '''
    DROP TABLE my_table;
    '''
    )

results = db.execute(raw_sql)
```