# Vector search queries

Vector search queries are built with the `.like` operator.
This allows developers to combine standard database with vector-search queries.
The philosophy is that developers do not need to convert their inputs 
into vector's themselves. Rather, this is taken care by the specified 
[`VectorIndex` component](../apply_api/vector_index).

The basic schematic for vector-search queries is:

```python
table_or_collection
    .like(Document(<dict-to-search-with>), vector_index='<my-vector-index>')      # the operand is vectorized using registered models
    .filter_results(*args, **kwargs)            # the results of vector-search are filtered
```

***or...***

```python
table_or_collection
    .filter_results(*args, **kwargs)            # the results of vector-search are filtered
    .like(Document(<dict-to-search-with>),
          vector_index='<my-vector-index>')      # the operand is vectorized using registered models
```

## MongoDB

```python
from superduperdb.ext.pillow import pil_image
from superduperdb import Document

my_image = PIL.Image.open('test/material/data/test_image.png')

q = db['my_collection'].find({'brand': 'Nike'}).like(Document({'img': pil_image(my_image)}), 
                                               vector_index='<my-vector-index>')

results = q.execute()
```

## SQL

```python
t = db['my_table']
t.filter(t.brand == 'Nike').like(Document({'img': pil_image(my_image)}))

results = db.execute(q)
```

