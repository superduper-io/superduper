import typing as t

from superduper import CFG
from superduper.base.annotations import trigger
from superduper.base.schema import Schema
from superduper.components.component import Component
from superduper.misc import typing as st  # noqa: F401
from superduper.misc.importing import import_object

DEFAULT_PRIMARY_ID = 'id'


class Table(Component):
    """
    A component that represents a table in a database.

    :param fields: The schema of the table
    :param primary_id: The primary id of the table
    :param data: Data to insert post creation
    :param path: The path to the class
    :param is_component: Whether the table is a component
    """

    fields: t.Dict | None = None
    primary_id: str = DEFAULT_PRIMARY_ID
    data: Component | None = None
    path: str | None = None
    is_component: bool = False
    component_cache: t.ClassVar[bool] = True

    def postinit(self):
        """Post initialization method."""
        if self.path is None:
            assert isinstance(self.fields, dict), "Fields must be set if cls is not set"
            self.schema = Schema.build(**self.fields)
            self.cls = None
        else:
            self.cls = import_object(self.path)
            self.schema = self.cls.class_schema
            self.fields = self.cls._new_fields
        super().postinit()

    def create_table_events(self):
        """Create the table events."""
        from superduper.base.event import CreateTable

        return {
            self.identifier: CreateTable(
                identifier=self.identifier,
                primary_id=self.primary_id,
                fields=self.fields,
                is_component=self.is_component,
            )
        }

    def cleanup(self):
        """Cleanup the table, on removal of the component."""
        if self.identifier.startswith(CFG.output_prefix):
            self.db.databackend.drop_table(self.identifier)

    @trigger('apply', requires='data')
    def add_data(self):
        if isinstance(self.data, Component):
            data = self.data.data
        else:
            data = self.data
        if data:
            self.db[self.identifier].insert(data)
