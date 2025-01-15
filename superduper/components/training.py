import typing as t

from superduper.components.component import Component


class Checkpoint(Component):
    """Checkpoint component for saving the model checkpoint.

    :param path: The path to the checkpoint.
    :param step: The step of the checkpoint.
    """

    _fields = {'path': 'file'}
    type_id: t.ClassVar[str] = "checkpoint"

    path: t.Optional[str]
    step: int

    def __post_init__(self, db):
        super().__post_init__(db)
        self.version = int(self.step)
