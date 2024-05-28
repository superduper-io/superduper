# `VectorIndex`

- Wrap a `Listener` so that outputs are searchable
- Can optionally take a second `Listener` for multimodal search
- Applies to `Listener` instances containing `Model` instances which output vectors, arrays or tensors
- Maybe leveraged in SuperDuperDB queries with the `.like` operator

***Dependencies***

- [`Listener`](./listener.md)

***Usage pattern***

```python
from superduperdb import VectorIndex

vi = VectorIndex(
    'my-index',
    indexing_listener=listener_1  # defined earlier calculates searchable vectors
)

# or directly from a model
vi = model_1.to_vector_index(select=q, key='x')

# or may use multiple listeners
vi = VectorIndex(
    'my-index',
    indexing_listener=listener_1
    compatible_listener=listener_2 # this listener can have `listener_2.active = False`
)

db.apply(vi)
```

***See also***

- [vector-search queries](../query_api/vector_search)
- [vector-search service](../cluster_mode/vector_comparison_service)