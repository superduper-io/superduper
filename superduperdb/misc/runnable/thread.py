import dataclasses as dc
import sys
import typing as t
from functools import cached_property, partial
from threading import Thread

from .runnable import Callback, Runnable

print_err = partial(print, file=sys.stderr)


class ThreadBase(Runnable):
    """A base class for classes with a thread.

    It adds the following features to threading.Thread:

    * Has Events `running` and `stopped` with `on_set` callbacks
    * Handles exceptions and prints or redirects them
    * Runs once, or multiple times, depending on `self.looping`
    """

    callback: Callback
    error: t.Callable[[Exception], None]
    daemon: bool = False
    looping: bool = False
    name: str = ''

    def __str__(self):
        return self.name or self.__class__.__name__

    def pre_run(self):
        pass

    def run(self):
        self.pre_run()
        self.running.set()

        while self.running:
            try:
                self.callback()
            except Exception as e:
                self.error(e)
                self.stop()
            else:
                if not self.looping:
                    break
        if self.running:
            self.running.clear()
        self.stopped.set()

    def stop(self):
        self.running.clear()

    def finish(self):
        pass


class IsThread(ThreadBase, Thread):
    """This ThreadBase inherits from threading.Thread.

    To use IsThread, derive from it and override either or both of
    self.callback() and self.pre_run()

    """

    def __init__(self, *args, **kwargs):
        ThreadBase.__init__(self, *args, **kwargs)
        Thread.__init__(self, daemon=self.daemon)

    def callback(self):
        pass

    def error(self, item: Exception) -> None:
        print_err(item)

    def join(self, timeout: t.Optional[float] = None):
        Thread.join(self, timeout)

    def start(self):
        Thread.start(self)


@dc.dataclass
class HasThread(ThreadBase):
    """This ThreadBase contains a thread, and is constructed with a callback"""

    callback: Callback = print
    daemon: bool = False
    error: t.Callable = print_err
    looping: bool = False
    name: str = ''

    def __post_init__(self):
        ThreadBase.__init__(self)

    def join(self, timeout: t.Optional[float] = None):
        self.thread.join(timeout)

    def start(self):
        self.thread.start()

    def new_thread(self) -> Thread:
        return Thread(target=self.run, daemon=self.daemon)

    @cached_property
    def thread(self) -> Thread:
        return self.new_thread()

    def __str__(self):
        return self.name or f'HasThread[{self.callback}]'
