from . import dataclasses as dc
import pickle
import typing as t


@dc.dataclass
class PickleObject:
    """
    JSONize an object by pickling

    The result is not JSONable, because it contains bytes.
    """

    object: dc.InitVar[t.Any] = None  # type: ignore[assignment]
    pickled: t.Optional[bytes] = None

    def __post_init__(self, object: t.Any):
        if object is None and self.pickled is not None:
            self.object = pickle.loads(self.pickled)
        else:
            self.object = object

        if self.pickled is None and object is not None:
            self.pickled = pickle.dumps(object)
