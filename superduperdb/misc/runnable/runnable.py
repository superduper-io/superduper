import threading
import typing as t

Callback = t.Callable[[], None]


class Event(threading.Event):
    """
    A threading.Event that also calls back to zero or more functions when its state
    is set or reset, and has a __bool__ method.

    Note that the callback might happen on some completely different thread,
    so these functions cannot block"""

    on_set: t.List[Callback]

    def __init__(self, *on_set: Callback):
        self.on_set = list(on_set)
        super().__init__()

    def set(self):
        super().set()
        [c() for c in self.on_set]

    def clear(self):
        super().clear()
        [c() for c in self.on_set]

    def __bool__(self):
        return self.is_set()


class Runnable:
    """A base class for things that start, run, finish, stop and join

    Stopping is requesting immediate termination: finishing is saying that
    there is no more work to be done, finish what you are doing.

    A Runnable has two `Event`s, `running` and `stopped`, and you can either
    `wait` on either of these conditions to be true, or add a callback function
    (which must be non-blocking) to either of them.

    `running` is not set until the setup for a `Runnable` has finished;
    `stopped` is not set until all the computations in a thread have ceased.

    An Runnable can be used as a context manager:

        with runnable:
            # The runnable is running by this point
            do_stuff()
        # By the time you get to here, the runnable has completely stopped

    The above means roughly the same as

        runnable.start()
        try:
            do_stuff()
            runnable.finish()
            runnable.join()
        finally:
            runnable.stop()

    """

    #: An Event that is only set once this object is actually running
    running: Event

    #: An event that is only set once this object is fully stopped
    stopped: Event

    def __init__(self):
        self.running = Event()
        self.stopped = Event()

    def start(self):
        """Start this object.

        Note that self.running might not be immediately true after this method completes
        """
        self.running.set()

    def stop(self):
        """Stop as soon as possible. might not do anything, should never raise.

        Note that self.stopped might not be immediately true after this method completes
        """
        self.running.clear()
        self.stopped.set()

    def finish(self):
        """Request an orderly shutdown where all existing work is completed.

        Note that self.stopped might not be immediately true after this method completes
        """
        self.stop()

    def join(self, timeout: t.Optional[float] = None):
        """Join this thread or process.  Might block indefinitely, might do nothing"""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                self.finish()
                self.join()
        finally:
            self.stop()
