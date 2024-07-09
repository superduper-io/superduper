**`superduper.vector_search.atlas`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper/vector_search/atlas.py)

## `MongoAtlasVectorSearcher` 

```python
MongoAtlasVectorSearcher(self,
     identifier: str,
     collection: str,
     dimensions: Optional[int] = None,
     measure: Optional[str] = None,
     output_path: Optional[str] = None)
```
| Parameter | Description |
|-----------|-------------|
| identifier | Unique string identifier of index |
| collection | Collection name |
| dimensions | Dimension of the vector embeddings |
| measure | measure to assess similarity |
| output_path | Path to the output |

Vector searcher implementation of atlas vector search.

