import dataclasses as dc
import time
import typing as t
from queue import Empty, Queue

from .runnable import Event


@dc.dataclass
class QueueChunker:
    """Chunk a queue into lists of length at most `chunk_size` within time `timeout`"""

    #: Maximum number of entries in a chunk
    chunk_size: int

    #: Maximum amount of time to block
    timeout: float

    #: If accumulate timeouts is True, then `timeout` is the total timeout
    #: allowed over the whole chunk, otherwise the timeout is applied to each item.
    accumulate_timeouts: bool = False

    def __call__(self, queue: Queue, stop_event: Event) -> t.Iterator[t.List]:
        def chunk():
            start = self.accumulate_timeouts and time.time()

            for i in range(self.chunk_size):
                if stop_event:
                    return

                elapsed = self.accumulate_timeouts and time.time() - start
                timeout = self.timeout - elapsed

                try:
                    item = queue.get(timeout=timeout)
                except Empty:
                    return
                else:
                    yield item

        while not stop_event:
            if c := list(chunk()):
                yield c
