import dataclasses as dc
import traceback
import typing as t
from functools import cached_property, partial, wraps
from threading import Thread

import superduper as s

from .runnable import Callback, Runnable


def none(x):
    """No-op function.

    :param x: Any
    """
    pass


def _debug(method=None, before=True, after=False):
    if method is None:
        return partial(_debug, before=before, after=after)

    @wraps(method)
    def wrapped(self, *a, **ka):
        msg = f'{self}: {method.__name__}'

        try:
            before and s.logging.debug(f'{msg}: before')
            return method(self, *a, **ka)
        finally:
            after and s.logging.debug(f'{msg}: after')

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
        """Pre-run the thread."""
        pass

    @_debug(after=True)
    def run(self):
        """Run the thread."""
        self.pre_run()
        self.running.set()

        while self.running:
            try:
                self.callback()
            except Exception as e:
                exc = traceback.format_exc()
                s.logging.error(f'{self}: Exception\n{exc}')

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
        """Stop the thread."""
        self.running.clear()

    @_debug
    def finish(self):
        """Finish the thread."""
        pass


@dc.dataclass
class HasThread(ThreadBase):
    """HasThread contains a thread, and is constructed with a callback.

    :param callback: The callback to run in the thread.
    :param daemon: Whether the thread is a daemon.
    :param error: The error callback.
    :param looping: Whether the thread should loop.
    :param name: The name of the thread.
    """

    callback: Callback = print
    daemon: bool = False
    error: t.Callable = none
    looping: bool = False
    name: str = ''

    def __post_init__(self):
        ThreadBase.__init__(self)

    @_debug(after=True)
    def join(self, timeout: t.Optional[float] = None):
        """Join the thread.

        :param timeout: Timeout in seconds
        """
        self.thread.join(timeout)

    @_debug
    def start(self):
        """Start the thread."""
        self.thread.start()

    def new_thread(self) -> Thread:
        """Return a new thread."""
        return Thread(target=self.run, daemon=self.daemon)

    @cached_property
    def thread(self) -> Thread:
        """Return the thread."""
        return self.new_thread()


@dc.dataclass
class CallbackLoopThread(HasThread):
    """HasThread contains a thread, and is constructed with a callback.

    :param callback: The callback to run in the thread.
    :param daemon: Whether the thread is a daemon.
    :param error: The error callback.
    :param looping: Whether the thread should loop.
    :param name: The name of the thread.
    """

    callback: Callback = print
    daemon: bool = False
    error: t.Callable = none
    looping: bool = False
    name: str = ''

    def __post_init__(self):
        super().__post_init__()

    @_debug(after=True)
    def join(self, timeout: t.Optional[float] = None):
        """Join the thread.

        :param timeout: Timeout in seconds
        """
        self.thread.join(timeout)

    @_debug
    def start(self):
        """Start the thread."""
        self.thread.start()

    def new_thread(self) -> Thread:
        """Return a new thread."""
        return Thread(target=self.run, daemon=self.daemon)

    @cached_property
    def thread(self) -> Thread:
        """Return the thread."""
        return self.new_thread()

    @_debug(after=True)
    def run(self):
        """Run the thread."""
        self.pre_run()
        self.running.set()

        try:
            self.callback(running=self.running)
        except Exception as e:
            exc = traceback.format_exc()
            s.logging.error(f'{self}: Exception\n{exc}')

            self.error(e)
            self.stop()
        if self.running:
            self.running.clear()
        self.stopped.set()
