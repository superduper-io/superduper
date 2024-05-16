import dataclasses as dc
import typing as t

from superduperdb import logging
from superduperdb.backends.ibis.field_types import dtype
from superduperdb.components.component import Component
from superduperdb.components.schema import Schema, _Native

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer

DEFAULT_PRIMARY_ID = 'id'


@dc.dataclass(kw_only=True)
class Table(Component):
    """
    A component that represents a table in a database.

    :param schema: The schema of the table
    :param primary_id: The primary id of the table
    """

    type_id: t.ClassVar[str] = 'table'
    schema: Schema
    primary_id: str = DEFAULT_PRIMARY_ID

    def __post_init__(self, db, artifacts):
        super().__post_init__(db, artifacts)
        fields = {}
        if '_fold' not in self.schema.fields:
            fields.update({'_fold': dtype('str')})

        if '_schema' not in self.schema.fields:
            fields.update({'_schema': dtype('str')})


        for k, v in self.schema.fields.items():
            if isinstance(v, _Native):
                fields[k] = dtype(v.identifier)
            else:
                fields[k] = v

        self.schema = Schema(
            self.schema.identifier,
            fields={**fields},
        )

        assert self.primary_id != '_input_id', '"_input_id" is a reserved value'

    def pre_create(self, db: 'Datalayer'):
        """
        Create the table in the database.

        :param db: The datalayer isinstance
        """
        assert self.schema is not None, "Schema must be set"
        # TODO why? This is done already
        for e in self.schema.encoders:
            db.add(e)
        if db.databackend.in_memory:
            logging.info(f'Using in-memory tables "{self.identifier}" so doing nothing')
            return

        try:
            db.databackend.create_table_and_schema(self.identifier, self.schema.raw)
        except Exception as e:
            if 'already exists' in str(e):
                pass
            else:
                raise e
