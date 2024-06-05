import typing as t

from superduperdb.components.component import Component


class Application(Component):
    """
    Application built from template.

    :param template: Template.
    :param kwargs: Keyword arguments passed to `template`.
    """

    type_id: t.ClassVar[str] = 'application'
    template: str = None
    kwargs: t.Dict

    def __post_init__(self, db, artifacts):
        return super().__post_init__(db, artifacts)

    def post_create(self, db):
        """
        Database `PostCreate` hook.

        :param db: Datalayer instance.
        """
        template = db.load('template', self.template)
        component = template(**self.kwargs)
        db.apply(component)
