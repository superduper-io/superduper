import threading
import typing as t

Callback = t.Callable[[], None]


class Event(threading.Event):
    """An Event that calls a list of callbacks when set or cleared.

    A threading.Event that also calls back to zero or more functions when its state
    is set or reset, and has a __bool__ method.

    Note that the callback might happen on some completely different thread,
    so these functions cannot block

    :param on_set: Callbacks to call when the event is set
    """

    on_set: t.List[Callback]

    def __init__(self, *on_set: Callback):
        self.on_set = list(on_set)
        super().__init__()

    def set(self):
        """Set the flag to True and call all the callbacks."""
        super().set()
        [c() for c in self.on_set]

    def clear(self):
        """Clear the flag to False and call all the callbacks."""
        super().clear()
        [c() for c in self.on_set]

    def __bool__(self):
        return self.is_set()
