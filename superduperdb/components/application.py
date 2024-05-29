import dataclasses as dc
import typing as t

from superduperdb import logging
from superduperdb.components.component import Component
from superduperdb.components.template import Template
from superduperdb.misc.annotations import merge_docstrings


@merge_docstrings
@dc.dataclass(kw_only=True)
class Application(Component):
    """
    Application built from template.

    :param template: Template.
    :param kwargs: Keyword arguments passed to `template`.
    """

    type_id: t.ClassVar[str] = 'application'

    template: t.Union[Template, str] = None
    kwargs: t.Dict

    def __post_init__(self, db, artifacts):
        if isinstance(self.template, str):
            self.template = db.load('template', self.template)
        self._component = None
        return super().__post_init__(db, artifacts)

    @property
    def component(self):
        """Application loaded component from template."""
        if self._component is None:
            logging.warn('Component is not yet loaded, apply this application to db')

        return self._component

    def post_create(self, db):
        """
        Database `PostCreate` hook.

        :param db: Datalayer instance.
        """
        component = self.template(**self.kwargs)
        self._component = component

        db.apply(component)
