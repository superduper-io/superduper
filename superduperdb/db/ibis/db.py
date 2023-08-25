from superduperdb.db.ibis.cursor import SuperDuperIbisCursor
from superduperdb.db.base.db import DB
from superduperdb.db.ibis.query import OutputTable
from superduperdb import logging


class IbisDB(DB):
    def _execute(self, query, parent):
        table = parent
        for member in query.members:
            parent = member.execute(
                self, parent, table=query.collection, ibis_table=table
            )
        cursor = SuperDuperIbisCursor(parent, query.collection.primary_id, encoders=self.encoders)
        return cursor.execute()

    def execute(self, query):
        if isinstance(query, SuperDuperIbisCursor):
            return query.execute()

        table = query.collection.get_table(self.db)
        return self._execute(query, table)
    
    def create_output_table(self, model):
        try:
            table = OutputTable(model=model)
            table.create(self.db)
        except Exception as e:
            logging.error(e)
        else:
            logging.debug(f"Created output table for {model}")
