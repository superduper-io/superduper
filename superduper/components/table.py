import typing as t

from superduper import CFG
from superduper.base.annotations import trigger
from superduper.base.schema import Schema
from superduper.components.component import Component
from superduper.misc import typing as st  # noqa: F401
from superduper.misc.importing import import_object

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer

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

    def cleanup(self):
        """Cleanup the table, on removal of the component."""
        self.db.databackend.drop_table(self.identifier)
        if self.db.cluster.cache is not None:
            from superduper import logging

            logging.info(f'Deleting schema for table {self.identifier}')
            del self.db.cluster.cache[f'Table/{self.identifier}/schema']
            logging.info(f'Deleting schema for table {self.identifier}... DONE')

    def on_create(self):
        """Create the table, on creation of the component."""
        assert self.schema is not None, "Schema must be set"

        try:
            self.db.metadata.create_table_and_schema(
                self.identifier,
                schema=self.schema,
                primary_id=self.primary_id,
                is_component=self.is_component,
            )
        except Exception as e:
            if 'already exists' in str(e):
                pass
            else:
                raise e

    @trigger('apply', requires='data')
    def add_data(self):
        if isinstance(self.data, Component):
            data = self.data.data
        else:
            data = self.data
        if data:
            self.db[self.identifier].insert(data)
