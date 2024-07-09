**`superduper.vector_search.in_memory`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper/vector_search/in_memory.py)

## `InMemoryVectorSearcher` 

```python
InMemoryVectorSearcher(self,
     identifier: str,
     dimensions: int,
     h: Optional[numpy.ndarray] = None,
     index: Optional[List[str]] = None,
     measure: Union[str,
     Callable] = 'cosine')
```
| Parameter | Description |
|-----------|-------------|
| identifier | Unique string identifier of index |
| dimensions | Dimension of the vector embeddings |
| h | array/ tensor of vectors |
| index | list of IDs |
| measure | measure to assess similarity |

Simple hash-set for looking up with vector similarity.

