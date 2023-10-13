from superduperdb import logging
from superduperdb.db.base.db import DB
from superduperdb.db.ibis.cursor import SuperDuperIbisCursor
from superduperdb.db.ibis.query import InMemoryTable, OutputTable

_INMEMORY_BACKENDS = ['duckdb']


class IbisDB(DB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._table = None
        if self.backend in _INMEMORY_BACKENDS:
            self._table = InMemoryTable(identifier=self.db.name, table=self.db)

    @property
    def backend(self):
        return self.databackend.backend

    @property
    def table(self):
        return self._table

    def _execute(self, query, parent):
        table = parent
        for member in query.members:
            parent = member.execute(
                self, parent, table=query.collection, ibis_table=table
            )
        cursor = SuperDuperIbisCursor(
            parent, query.collection.primary_id, encoders=self.encoders
        )
        return cursor.execute()

    def execute(self, query):
        if isinstance(query, SuperDuperIbisCursor):
            return query.execute()

        if self.backend in _INMEMORY_BACKENDS:
            table = self.db
        else:
            table = query.collection.get_table(self.db)
        return self._execute(query, table)

    def create_output_table(self, model):
        try:
            table = OutputTable(model=model.identifier, output_type=model.encoder)
            table.create(self.db)
        except Exception as e:
            logging.error(e)
        else:
            logging.debug(f"Created output table for {model}")
