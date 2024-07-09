import typing as t

from .runnable import Runnable

_SENTINEL_MESSAGE = object()


class HasRunnables(Runnable):
    """Collect zero or more Runnable into one."""

    runnables: t.Sequence[Runnable]

    def start(self):
        """Start all runnables."""
        for r in self.runnables:
            r.running.on_set.append(self._on_start)
            r.stopped.on_set.append(self._on_stop)
            r.start()

    def stop(self):
        """Stop all runnables."""
        self.running.clear()
        for r in self.runnables:
            r.stop()

    def finish(self):
        """Finish all runnables."""
        for r in self.runnables:
            r.finish()

    def join(self, timeout: t.Optional[float] = None):
        """Join all runnables.

        :param timeout: Timeout in seconds
        """
        for r in self.runnables:
            r.join(timeout)

    def _on_start(self):
        if not self.running and all(r.running for r in self.runnables):
            super().start()

    def _on_stop(self):
        if not self.stopped and all(r.stopped for r in self.runnables):
            super().stop()
