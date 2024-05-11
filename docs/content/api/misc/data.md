**`superduperdb.misc.data`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/misc/data.py)

## `ibatch` 

```python
ibatch(iterable: Iterable[~T],
     batch_size: int) -> Iterator[List[~T]]
```
| Parameter | Description |
|-----------|-------------|
| iterable | the iterable to batch |
| batch_size | the number of groups to write |

Batch an iterable into chunks of size `batch_size`.

