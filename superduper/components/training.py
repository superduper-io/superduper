import typing as t

from superduper.components.component import Component
from superduper.components.datatype import DataType, file_lazy


class Checkpoint(Component):
    """Checkpoint component for saving the model checkpoint.

    :param path: The path to the checkpoint.
    :param step: The step of the checkpoint.
    """

    path: t.Optional[str]
    step: int
    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, DataType]]] = (("path", file_lazy),)
    type_id: t.ClassVar[str] = "checkpoint"

    def __post_init__(self, db, artifacts):
        super().__post_init__(db, artifacts)
        self.version = int(self.step)
