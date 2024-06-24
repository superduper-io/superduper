# Create vector-index


```python
vector_index_name = 'my-vector-index'
```


```python
# <tab: 1-Modality>
from superduperdb import VectorIndex, Listener

jobs, _ = db.add(
    VectorIndex(
        vector_index_name,
        indexing_listener=Listener(
            key=indexing_key,      # the `Document` key `model` should ingest to create embedding
            select=select,       # a `Select` query telling which data to search over
            model=embedding_model,         # a `_Predictor` how to convert data to embeddings
        )
    )
)
```


```python
# <tab: 2-Modalities>
from superduperdb import VectorIndex, Listener

jobs, _ = db.add(
    VectorIndex(
        vector_index_name,
        indexing_listener=Listener(
            key=indexing_key,      # the `Document` key `model` should ingest to create embedding
            select=select,       # a `Select` query telling which data to search over
            model=embedding_model,         # a `_Predictor` how to convert data to embeddings
        ),
        compatible_listener=Listener(
            key=compatible_key,      # the `Document` key `model` should ingest to create embedding
            model=compatible_model,         # a `_Predictor` how to convert data to embeddings
            active=False,
            select=None,
        )
    )
)
```


```python
query_table_or_collection = select.table_or_collection
```
