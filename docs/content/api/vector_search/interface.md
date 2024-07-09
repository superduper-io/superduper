**`superduper.vector_search.interface`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper/vector_search/interface.py)

## `FastVectorSearcher` 

```python
FastVectorSearcher(self,
     db: 'Datalayer',
     vector_searcher,
     vector_index: str)
```
| Parameter | Description |
|-----------|-------------|
| db | Datalayer instance |
| vector_searcher | Vector searcher instance |
| vector_index | Vector index name |

Fast vector searcher implementation using the server.

