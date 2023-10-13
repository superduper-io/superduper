from warnings import warn

import click
import ibis
from ibis.backends.base import BaseBackend

from superduperdb.db.base.data_backend import BaseDataBackend
from superduperdb.misc.colors import Colors


class IbisDataBackend(BaseDataBackend):
    def __init__(self, conn: BaseBackend, name: str):
        super().__init__(conn=conn, name=name)

        self._db = conn
        if isinstance(conn, BaseBackend):
            self._backend = "sql"
        elif isinstance(conn.op(), ibis.expr.operations.relations.InMemoryTable):
            # TODO: find the backend for in-memory tables
            self._backend = "duckdb"

    def build_artifact_store(self):
        raise NotImplementedError

    def build_metadata(self):
        raise NotImplementedError

    @property
    def backend(self):
        return self._backend

    @property
    def db(self):
        return self._db

    def create_table_and_schema(self, identifier: str, mapping: dict):
        """
        Create a schema in the data-backend.
        """
        try:
            t = self.conn.create_table(identifier, schema=ibis.schema(mapping))
        except Exception as e:
            if 'exists' in str(e):
                warn("Table already exists, skipping...")
                t = self.conn.table(identifier)
            else:
                raise e
        return t

    def drop(self, force: bool = False):
        if not force:
            if not click.confirm(
                f'{Colors.RED}[!!!WARNING USE WITH CAUTION AS YOU '
                f'WILL LOSE ALL DATA!!!]{Colors.RESET} '
                'Are you sure you want to drop the data-backend? ',
                default=False,
            ):
                print('Aborting...')
        # TODO: Implement drop functionality
        return
