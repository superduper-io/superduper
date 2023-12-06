import dataclasses as dc
import typing as t

from superduperdb.base.artifact import Artifact
from superduperdb.components.component import Component
from superduperdb.misc.annotations import public_api


@public_api(stability='beta')
@dc.dataclass(kw_only=True)
class Metric(Component):
    """
    Metric base object with which to evaluate performance on a data-set.
    These objects are ``callable`` and are applied row-wise to the data, and averaged.
    {component_parameters}
    :param object: callable or ``Artifact`` to be applied to the data
    """

    __doc__ = __doc__.format(component_parameters=Component.__doc__)

    artifacts: t.ClassVar[t.List[str]] = ['object']
    type_id: t.ClassVar[str] = 'metric'

    object: t.Union[Artifact, t.Callable]

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.object and not isinstance(self.object, Artifact):
            self.object = Artifact(artifact=self.object)

    def __call__(self, x: t.Sequence[int], y: t.Sequence[int]) -> bool:
        assert isinstance(self.object, Artifact)
        return self.object.artifact(x, y)
