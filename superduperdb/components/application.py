import dataclasses as dc
import typing as t

from superduperdb.base.datalayer import Datalayer
from superduperdb.components.component import Component
from superduperdb.components.template import Template


@dc.dataclass(kw_only=True)
class Application(Component):
    """
    Application built from template.

    :param template: Template.
    :param kwargs: Keyword arguments passed to `template`.
    """
    type_id: t.ClassVar[str] = 'application'

    template: t.Union[Template, str] = None
    component: t.Optional[Component] = None
    kwargs: t.Dict

    def __post_init__(self, db, artifacts):
        if self.component is None:
            self.component = self.template(**self.kwargs)
            self.component.db = None
            self.template = self.template.identifier
        return super().__post_init__(db, artifacts)
