**`superduperdb.backends.query_dataset`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/backends/query_dataset.py)

## `query_dataset_factory` 

```python
query_dataset_factory(**kwargs)
```
| Parameter | Description |
|-----------|-------------|
| kwargs | Keyword arguments to be passed to the query dataset object. |

Create a query dataset object.

If ``data_prefetch`` is set to ``True``, then a ``CachedQueryDataset`` object is
created, otherwise a ``QueryDataset`` object is created.

## `CachedQueryDataset` 

```python
CachedQueryDataset(self,
     select: superduperdb.backends.base.query.Query,
     mapping: Optional[ForwardRef('Mapping')] = None,
     ids: Optional[List[str]] = None,
     fold: Optional[str] = 'train',
     transform: Optional[Callable] = None,
     db=None,
     in_memory: bool = True,
     prefetch_size: int = 100)
```
| Parameter | Description |
|-----------|-------------|
| select | A select query object which defines the query to be executed. |
| mapping | A mapping object to be used for the dataset. |
| ids | A list of ids to be used for the dataset. |
| fold | The fold to be used for the dataset. |
| transform | A callable which can be used to transform the dataset. |
| db | A datalayer instance to be used for the dataset. |
| in_memory | A boolean flag to indicate if the dataset should be loaded |
| prefetch_size | The number of documents to prefetch from the database. |

Cached Query Dataset for fetching documents from database.

This class which fetch the document corresponding to the given ``index``.
This class prefetches documents from database and stores in the memory.

This can drastically reduce database read operations and hence reduce the overall
load on the database.

## `ExpiryCache` 

```python
ExpiryCache(self,
     /,
     *args,
     **kwargs)
```
| Parameter | Description |
|-----------|-------------|
| args | *args for `list` |
| kwargs | **kwargs for `list` |

Expiry Cache for storing documents.

The document will be removed from the cache after fetching it from the cache.

## `QueryDataset` 

```python
QueryDataset(self,
     select: superduperdb.backends.base.query.Query,
     mapping: Optional[ForwardRef('Mapping')] = None,
     ids: Optional[List[str]] = None,
     fold: Optional[str] = 'train',
     transform: Optional[Callable] = None,
     db: Optional[ForwardRef('Datalayer')] = None,
     in_memory: bool = True)
```
| Parameter | Description |
|-----------|-------------|
| select | A select query object which defines the query to be executed. |
| mapping | A mapping object to be used for the dataset. |
| ids | A list of ids to be used for the dataset. |
| fold | The fold to be used for the dataset. |
| transform | A callable which can be used to transform the dataset. |
| db | A datalayer instance to be used for the dataset. |
| in_memory | A boolean flag to indicate if the dataset should be loaded in memory. |

Query Dataset for fetching documents from database.

