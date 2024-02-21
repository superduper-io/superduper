# Vector-Search on Snowflake

In this use-case we describe how to implement vector-search using `superduperdb` on Snowflake.

Here is the notebook [notebook](https://github.com/SuperDuperDB/superduperdb/blob/main/examples/snowflake-example.ipynb) for you to run.

### Configure `superduperdb` to work with Snowflake

The first step in doing this is 
to connect to your snowflake account. When you log in it should look something like this:

![](/img/snowflake-login.png)

The important thing you need to get from this login is the **organization-id** and **user-id** from the menu in the bottom right (annotated on the image). You will set these values in the cell below.



```python
import os

# We set this value, since Snowflake via `ibis` doesn't support `bytes` directly.
os.environ['SUPERDUPERDB_BYTES_ENCODING'] = 'Str'

from superduperdb import superduper, CFG

user = "<USERNAME>"
password = "<PASSWORD>"
account = "WSWZPKW-LN66790"  # ORGANIZATIONID-USERID

def make_uri(database):
    return f"snowflake://{user}:{password}@{account}/{database}"
```

## Set up sample data to test vector-search

We're going to use some of the Snowflake sample data in this example, namely the `FREE_COMPANY_DATASET`. You 
can find the `FREE_COMPANY_DATASET` on [this link](https://app.snowflake.com/marketplace/listing/GZSTZRRVYL2/people-data-labs-free-company-dataset).

Since the database where this data is hosted is read-only, we copy a sample of the data to our own dataset, and work with the data there.


```python
from superduperdb.backends.ibis.query import RawSQL

db = superduper(
    make_uri("FREE_COMPANY_DATASET/PUBLIC"),
    metadata_store='sqlite:///.testdb.db'
)

sample = db.execute(RawSQL('SELECT * FROM FREECOMPANYDATASET SAMPLE (5000 ROWS);')).as_pandas()
```

### Connect to your dedicated vector-search database

We use the connection we created to get the snapshot, to also create the dataset we are going to work with:


```python
db.databackend.conn.create_database('SUPERDUPERDB_EXAMPLE', force=True)
```

Now we are ready to connect to this database with `superduperdb`:


```python
from superduperdb.backends.ibis.query import RawSQL

db = superduper(
    make_uri("SUPERDUPERDB_EXAMPLE/PUBLIC"),
    metadata_store='sqlite:///.testdb.db'
)
```

Since `superduperdb` implements extra features on top of your classical database/ datalake, it's necessary
to add the tables you wish to work with to the system. You'll notice we are creating a schema as well; that allows
us to implement "interesting" data-types on top of Snowflake, such as images or audio.


```python
from superduperdb.backends.ibis.query import Table
from superduperdb.backends.ibis.field_types import dtype
from superduperdb import Schema

_, t = db.add(
    Table(
        'MYTABLE',
        primary_id='ID',
        schema=Schema(
            'MYSCHEMA',
            fields={
                k: dtype('str') 
                for k in sample.columns
            }
        )
    )
)
```

If you log back into Snowflake now it should look like this:

![](/img/snowflake-table.png)

You'll see that the database and table have been created.

### Insert data into the vector-search table

Before inserting the data, we'll do a little bit of cleaning. The reason for this, is we want to have clean ids which uniquely define 
the rows which we are working with:


```python
import random
sample.ID = sample.ID.str.replace('[^A-Za-z0-9\-]', '', regex=True).str.replace('[-]+', '-', regex=True)
sample[sample.isnull()] = None
del sample['FOUNDED']

random_id = lambda: str(random.randint(100000, 999999))
sample.ID = sample.ID.apply(lambda x: x + '-' + random_id())
```


```python
t = db.load('table', 'MYTABLE')
t
```

Now that we've created the table we want to work with, we can insert the sample data


```python
db.execute(t.insert(sample))
```

Let's check this was successful by fetching some data:


```python
list(db.execute(t.limit(5)))
```

In the next step, we're going to port a model from `sentence_transformers` to `superduperdb` and use this model in searching the rows 
of the table with vector-search. You can see that, in-addition to the `sentence_transformers` model, `superduperdb` allows
developers to specify a preprocessing (and postprocessing) function in their `Model` instances. In this case, 
the `preprocess` argument tells the model how to convert with the individual lines of data (dictionaries) to strings, so that the model can understand these lines:


```python
from superduperdb.ext.sentence_transformers import SentenceTransformer
from superduperdb.ext.numpy import array

model = SentenceTransformer(
    identifier='all-MiniLM-L6-v2',
    preprocess=lambda r: '; '.join([f'{k}={v}' for k, v in r.items()]),
    encoder=array(dtype='float32', shape=(384,)),
    predict_method='encode',
    batch_predict=True,
    device='mps',
)
```

This model is then sent to the `VectorIndex` component via `Listener` and registered with the system
with `db.add`:


```python
from superduperdb import Listener, VectorIndex

db.add(
    VectorIndex(
        identifier='my-index',
        indexing_listener=Listener(
            select=t,
            key='_base',
            model=model,
            predict_kwargs={'max_chunk_size': 500, 'batch_size': 30},
        ),
    )
)
```

This step will take a few moments (unless you have a GPU to hand).

:::important
**Once this step is finished you can 
search Snowflake with vector-search!**
:::

### Execute a vector-search query with `.like`

```python
from superduperdb import Document

cur = db.execute(
    t
        .like(Document({'description': 'A management consultancy company based in the USA'}), vector_index='my-index', n = 3)
        .limit(3)
)
```

We can view the results as a dataframe:


```python
cur.as_pandas()
```
