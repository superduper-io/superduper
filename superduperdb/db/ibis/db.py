import typing as t

from superduperdb.db.ibis.cursor import SuperDuperIbisCursor
from superduperdb.db.base.db import DB
from superduperdb.container.component import Component
from superduperdb.container.job import Job
from superduperdb.db.ibis.query import OutputTable
from superduperdb import logging

class IbisDB(DB):
    def _execute(self, query, parent):
        table = parent
        for member in query.members:
            parent = member.execute(
                self, parent, table=query.collection, ibis_table=table
            )
        cursor = SuperDuperIbisCursor(parent, query.collection.primary_id, encoders={})
        return cursor.execute()

    def execute(self, query):
        table = query.collection.get_table(self.db)
        return self._execute(query, table)

    def add(self, 
        object: Component,
            dependencies: t.Sequence[Job] = ()):
        super().add(object, dependencies=dependencies)

        try:
            table = OutputTable(model=object.identifier)
            table.create(self.db)
        except Exception as e:
            logging.error(e)
        else:
            logging.debug(f"Created output table for {object.identifier}")
