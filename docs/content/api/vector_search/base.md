**`superduperdb.vector_search.base`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/vector_search/base.py)

## `cosine` 

```python
cosine(x,
     y)
```
| Parameter | Description |
|-----------|-------------|
| x | numpy.ndarray |
| y | numpy.ndarray, y should be normalized! |

Cosine similarity function for vector search.

## `dot` 

```python
dot(x,
     y)
```
| Parameter | Description |
|-----------|-------------|
| x | numpy.ndarray |
| y | numpy.ndarray |

Dot function for vector similarity search.

## `l2` 

```python
l2(x,
     y)
```
| Parameter | Description |
|-----------|-------------|
| x | numpy.ndarray |
| y | numpy.ndarray |

L2 function for vector similarity search.

## `BaseVectorSearcher` 

```python
BaseVectorSearcher(self,
     identifier: 'str',
     dimensions: 'int',
     h: 't.Optional[numpy.ndarray]' = None,
     index: 't.Optional[t.List[str]]' = None,
     measure: 't.Optional[str]' = None)
```
| Parameter | Description |
|-----------|-------------|
| identifier | Unique string identifier of index |
| dimensions | Dimension of the vector embeddings |
| h | Seed vectors ``numpy.ndarray`` |
| index | list of IDs |
| measure | measure to assess similarity |

Base class for vector searchers.

## `VectorItem` 

```python
VectorItem(self,
     id: 'str',
     vector: 'numpy.ndarray') -> None
```
| Parameter | Description |
|-----------|-------------|
| id | ID of the vector |
| vector | Vector of the item |

Class for representing a vector in vector search with id and vector.

## `VectorSearchConfig` 

```python
VectorSearchConfig(self,
     id: 'str',
     dimensions: 'int',
     measure: 'VectorIndexMeasureType' = <VectorIndexMeasureType.l2: 'l2'>,
     parameters: 't.Mapping[str,
     t.Any]' = <factory>) -> None
```
| Parameter | Description |
|-----------|-------------|
| id | Identifier for the vector index |
| dimensions | Dimensions of the vector |
| measure | Measure for vector search |
| parameters | Additional parameters for vector search |

Represents search config which helps initiate a vector searcher class.

## `VectorSearchResult` 

```python
VectorSearchResult(self,
     id: 'str',
     score: 'float') -> None
```
| Parameter | Description |
|-----------|-------------|
| id | ID of the vector |
| score | Similarity score of the vector |

Dataclass for representing vector search results with `id` and `score`.

