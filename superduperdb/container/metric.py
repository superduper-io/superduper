import dataclasses as dc
import typing as t

from superduperdb.container.artifact import Artifact
from superduperdb.container.component import Component


@dc.dataclass
class Metric(Component):
    """
    Metric base object with which to evaluate performance on a data-set.
    These objects are ``callable`` and are applied row-wise to the data, and averaged.
    """

    artifacts: t.ClassVar[t.List[str]] = ['object']

    identifier: str
    object: t.Union[Artifact, t.Callable, None] = None
    version: t.Optional[int] = None

    #: A unique name for the class
    type_id: t.ClassVar[str] = 'metric'

    def __post_init__(self) -> None:
        if self.object and not isinstance(self.object, Artifact):
            self.object = Artifact(artifact=self.object)

    def __call__(self, x: int, y: int) -> bool:
        return self.object.artifact(x, y)  # type: ignore[union-attr]
