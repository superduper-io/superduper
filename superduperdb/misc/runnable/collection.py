import dataclasses as dc
import typing as t
from functools import cached_property
from queue import Empty, Queue

from .runnable import Runnable
from .thread import HasThread, none

_SENTINEL_MESSAGE = object()


class HasRunnables(Runnable):
    """Collect zero or more Runnable into one"""

    runnables: t.Sequence[Runnable]

    def start(self):
        for r in self.runnables:
            r.running.on_set.append(self._on_start)
            r.stopped.on_set.append(self._on_stop)
            r.start()

    def stop(self):
        self.running.clear()
        for r in self.runnables:
            r.stop()

    def finish(self):
        for r in self.runnables:
            r.finish()

    def join(self, timeout: t.Optional[float] = None):
        for r in self.runnables:
            r.join(timeout)

    def _on_start(self):
        if not self.running and all(r.running for r in self.runnables):
            super().start()

    def _on_stop(self):
        if not self.stopped and all(r.stopped for r in self.runnables):
            super().stop()


@dc.dataclass
class ThreadQueue(HasRunnables):
    """A simple multi-producer, multi-consumer queue with one thread per consumer.

    There is a special `finish_message` value, which when received shuts down
    that consumer.  ThreadQueue.finish() puts one `self.finish_message` onto the
    queue for each consumer.
    """

    callback: t.Callable[[t.Any], None]
    error: t.Callable = none
    maxsize: int = 0
    name: str = 'thread_queue'
    thread_count: int = 1
    timeout: t.Optional[float] = 0.1

    def __post_init__(self):
        HasRunnables.__init__(self)
        self.runnables = tuple(self._thread(i) for i in range(self.thread_count))

    @cached_property
    def queue(self) -> Queue:
        return Queue(self.maxsize)

    def finish(self) -> None:
        """Put an empty message into the queue for each listener"""
        for _ in self.runnables:
            self.queue.put(_SENTINEL_MESSAGE)

    def _thread(self, i: int) -> HasThread:
        thread = HasThread(name=f'{self.name}-{i}', error=self.error)

        def callback():
            self.running.wait()
            while self.running and thread.running:
                try:
                    item = self.queue.get(timeout=self.timeout)
                except Empty:
                    continue
                if item is _SENTINEL_MESSAGE:
                    return
                try:
                    self.callback(item)
                except Exception:
                    self.stop()
                    raise

        thread.callback = callback
        return thread
