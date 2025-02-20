import typing as t

from superduper.components.component import Component


class Checkpoint(Component):
    """Checkpoint component for saving the model checkpoint.

    :param path: The path to the checkpoint.
    :param step: The step of the checkpoint.
    """

    path: t.Optional[str]
    step: int

    def postinit(self):
        """Post initialization method."""
        super().postinit()
        self.version = int(self.step)
