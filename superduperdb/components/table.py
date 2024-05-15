import dataclasses as dc
import typing as t

from superduperdb.components.component import Component
from superduperdb.components.schema import Schema
from superduperdb.backends.ibis.field_types import dtype
from superduperdb import logging

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer

DEFAULT_PRIMARY_ID = 'id'


@dc.dataclass(kw_only=True)
class Table(Component):
    type_id: t.ClassVar[str] = 'table'
    schema: Schema
    primary_id: str = DEFAULT_PRIMARY_ID

    def __post_init__(self, db, artifacts):
        super().__post_init__(db, artifacts)
        if '_fold' not in self.schema.fields:
            self.schema = Schema(
                self.schema.identifier,
                fields={**self.schema.fields, '_fold': dtype('str')},
            )

        assert self.primary_id != '_input_id', '"_input_id" is a reserved value'

    def pre_create(self, db: 'Datalayer'):
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
