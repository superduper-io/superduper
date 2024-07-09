**`superduper.vector_search.lance`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper/vector_search/lance.py)

## `LanceVectorSearcher` 

```python
LanceVectorSearcher(self,
     identifier: str,
     dimensions: int,
     h: Optional[numpy.ndarray] = None,
     index: Optional[List[str]] = None,
     measure: Optional[str] = None)
```
| Parameter | Description |
|-----------|-------------|
| identifier | Unique string identifier of index |
| dimensions | Dimension of the vector embeddings in the Lance dataset |
| h | Seed vectors ``numpy.ndarray`` |
| index | list of IDs |
| measure | measure to assess similarity |

Implementation of a vector index using the ``lance`` library.

