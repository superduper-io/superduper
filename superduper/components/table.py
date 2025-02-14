import typing as t

from superduper import CFG
from superduper.base.annotations import trigger
from superduper.components.component import Component
from superduper.components.schema import Schema
from superduper.misc import typing as st

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer

DEFAULT_PRIMARY_ID = 'id'


class Table(Component):
    """
    A component that represents a table in a database.

    :param fields: The schema of the table
    :param primary_id: The primary id of the table
    :param data: Data to insert post creation
    :param cls: The class to use for the schema
    :param path: The path to the class
    :param component: Whether the table is a component
    """

    fields: t.Dict | None = None
    primary_id: str = DEFAULT_PRIMARY_ID
    data: Component | None = None
    cls: st.PickleEncoder = None
    path: str | None = None
    component: bool = False

    def postinit(self):
        """Post initialization method."""
        if self.cls is None:
            assert isinstance(self.fields, dict), "Fields must be set if cls is not set"
            self.schema = Schema.build(self.fields)
        else:
            if self.path is None:
                self.path = self.cls.__module__ + '.' + self.cls.__name__
            self.schema = self.cls.class_schema
        super().postinit()

    def cleanup(self, db):
        """Cleanup the table, on removal of the component.

        :param db: The Datalayer instance
        """
        if self.identifier.startswith(CFG.output_prefix):
            db.databackend.drop_table(self.identifier)

    def on_create(self, db: 'Datalayer'):
        """Create the table, on creation of the component.

        :param db: The Datalayer instance
        """
        assert self.schema is not None, "Schema must be set"
        # TODO drop?
        if db.databackend.in_memory:
            if self.identifier.startswith(CFG.output_prefix):
                db.databackend.in_memory_tables[self.identifier] = (
                    db.databackend.create_table_and_schema(self.identifier, self.schema)
                )

        try:
            db.databackend.create_table_and_schema(self.identifier, self.schema)
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
