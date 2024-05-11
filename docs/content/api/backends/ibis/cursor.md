**`superduperdb.backends.ibis.cursor`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/backends/ibis/cursor.py)

## `SuperDuperIbisResult` 

```python
SuperDuperIbisResult(self,
     raw_cursor: Any,
     id_field: str,
     db: Optional[ForwardRef('Datalayer')] = None,
     scores: Optional[Dict[str,
     float]] = None,
     decode_function: Optional[Callable] = None,
     _it: int = 0) -> None
```
SuperDuperIbisResult class for ibis query results.

SuperDuperIbisResult represents ibis query results with options
to unroll results as i.e pandas.

