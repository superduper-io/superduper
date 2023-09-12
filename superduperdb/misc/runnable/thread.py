import dataclasses as dc
import traceback
import typing as t
from functools import cached_property, partial, wraps
from threading import Thread

import superduperdb as s

from .runnable import Callback, Runnable


def none(x):
    pass


def _debug(method=None, before=True, after=False):
    if method is None:
        return partial(_debug, before=before, after=after)

    @wraps(method)
    def wrapped(self, *a, **ka):
        msg = f'{self}: {method.__name__}'

        try:
            before and s.log.debug(f'{msg}: before')
            return method(self, *a, **ka)
        finally:
            after and s.log.debug(f'{msg}: after')

    return wrapped


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
        return f'({self.__class__.__name__}){self.name}'

    @_debug
    def pre_run(self):
        pass

    @_debug(after=True)
    def run(self):
        self.pre_run()
        self.running.set()

        while self.running:
            try:
                self.callback()
            except Exception as e:
                exc = traceback.format_exc()
                s.log.error(f'{self}: Exception\n{exc}')

                self.error(e)
                self.stop()
            else:
                if not self.looping:
                    break
        if self.running:
            self.running.clear()
        self.stopped.set()

    @_debug
    def stop(self):
        self.running.clear()

    @_debug
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
        pass

    @_debug(after=True)
    def join(self, timeout: t.Optional[float] = None):
        Thread.join(self, timeout)

    @_debug
    def start(self):
        Thread.start(self)


@dc.dataclass
class HasThread(ThreadBase):
    """This ThreadBase contains a thread, and is constructed with a callback"""

    callback: Callback = print
    daemon: bool = False
    error: t.Callable = none
    looping: bool = False
    name: str = ''

    def __post_init__(self):
        ThreadBase.__init__(self)

    @_debug(after=True)
    def join(self, timeout: t.Optional[float] = None):
        self.thread.join(timeout)

    @_debug
    def start(self):
        self.thread.start()

    def new_thread(self) -> Thread:
        return Thread(target=self.run, daemon=self.daemon)

    @cached_property
    def thread(self) -> Thread:
        return self.new_thread()
