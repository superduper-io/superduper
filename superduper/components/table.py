import typing as t

from superduper import CFG
from superduper.base.annotations import trigger
from superduper.components.component import Component
from superduper.components.schema import Schema

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer

DEFAULT_PRIMARY_ID = 'id'


class Table(Component):
    """
    A component that represents a table in a database.

    :param fields: The schema of the table
    :param primary_id: The primary id of the table
    :param data: Data to insert post creation
    """

    type_id: t.ClassVar[str] = 'table'

    fields: t.Dict
    primary_id: str = DEFAULT_PRIMARY_ID
    data: Component | None = None

    def postinit(self):
        """Post initialization method."""
        fields = {**self.fields, '_fold': 'str'}
        from superduper.components.datatype import INBUILT_DATATYPES

        fields = {k: INBUILT_DATATYPES[fields[k]] for k in fields}
        self.schema = Schema(fields)
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
