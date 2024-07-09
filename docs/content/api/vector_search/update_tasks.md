**`superduper.vector_search.update_tasks`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper/vector_search/update_tasks.py)

## `copy_vectors` 

```python
copy_vectors(vector_index: str,
     query: Union[Dict,
     superduper.backends.base.query.Query],
     ids: Sequence[str],
     db=typing.Optional[ForwardRef('Datalayer')])
```
| Parameter | Description |
|-----------|-------------|
| vector_index | A identifier of the vector-index. |
| query | A query which was used by `db._build_task_workflow` method |
| ids | List of ids which were observed as added/updated documents. |
| db | Datalayer instance. |

Copy vectors of a ``VectorIndex`` component from the databackend to the fast_vector_search backend.

## `delete_vectors` 

```python
delete_vectors(vector_index: str,
     ids: Sequence[str],
     db=typing.Optional[ForwardRef('Datalayer')])
```
| Parameter | Description |
|-----------|-------------|
| vector_index | A identifier of vector-index. |
| ids | List of ids which were observed as deleted documents. |
| db | Datalayer instance. |

Delete vectors of a ``VectorIndex`` component in the fast_vector_search backend.

