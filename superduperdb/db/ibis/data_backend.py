import typing as t
import click

from ibis.backends.base import BaseBackend
import ibis
from superduperdb.db.base.data_backend import BaseDataBackend


class IbisDatabackend(BaseDataBackend):
    def __init__(self, conn: t.Optional[BaseBackend] = None, type: str = 'sqlite', name: t.Optional[str] = None):
        if conn is not None:
            self.conn = conn
        else:
            self.conn = getattr(ibis, type).connect(name)

    def create_table_or_collection(self, name: str, schema: t.List):
        schema = ibis.schema(schema.dict())
        self.conn.create_table(name, schema=schema)
    
    def drop(self, force: bool = False):
        """
        Drop the databackend.
        """
        if not force and not click.confirm(f'Drop databackend {self.name}?', default=False):
            print('Aborting')
            return
        for table in self.conn.list_tables():
            self.conn.drop_table(table)