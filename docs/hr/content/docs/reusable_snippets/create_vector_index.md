# Create vector-index

- `model` is a `_Predictor` which converts input data to `vector`, `array` or `tensor`.
- `select` is a `Select` query telling which data to search over.
- `key` is a `str`, tuple of `str` or `dict` telling the models how to consume the content of `Document` instances.


```python
from superduperdb import VectorIndex, Listener

db.add(
    VectorIndex(
        'my-vector-index',
        listener=Listener(
            key='<my_key>',      # the `Document` key `model` should ingest to create embedding
            select=select,       # a `Select` query telling which data to search over
            model=model,         # a `_Predictor` how to convert data to embeddings
        )
    )
)
```
