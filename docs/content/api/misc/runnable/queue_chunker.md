**`superduperdb.misc.runnable.queue_chunker`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/misc/runnable/queue_chunker.py)

## `QueueChunker` 

```python
QueueChunker(self,
     chunk_size: int,
     timeout: float,
     accumulate_timeouts: bool = False) -> None
```
| Parameter | Description |
|-----------|-------------|
| chunk_size | Maximum number of entries in a chunk |
| timeout | Maximum amount of time to block |
| accumulate_timeouts | If accumulate timeouts is True, then `timeout` is the total timeout allowed over the whole chunk, otherwise the timeout is applied to each item. |

Chunk a queue into lists of length at most `chunk_size` within time `timeout`.

