import json
import textwrap
import time
import traceback
import typing as t
import uuid

import click
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from superduper_postgres.query import map_superduper_query_to_postgres_query
from superduper_postgres.schema import superduper_to_postgres_schema
from superduper import CFG, logging
from superduper.backends.base.data_backend import BaseDataBackend
from superduper.base.event import CreateTable
from superduper.base.query import Query
from superduper.base.schema import Schema

# Hybrid tables are a feature of Snowflake which are a proxy
# for transactional tables.
CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS "{identifier}" (
    {schema}
);
"""

INSERT = """
INSERT INTO "{table}" ({columns}) 
  VALUES %s
"""

REPLACE = """
INSERT INTO "{table}" ({columns})
VALUES %s
ON CONFLICT ({primary_id})
DO UPDATE SET
    {excluded_columns}
"""

UPDATE = """
UPDATE "{table}" SET {updates} WHERE {conditions}
"""

class PostgresDataBackend(BaseDataBackend):
    """Postgres data backend."""

    tables_ignore = '^POSTGRES_*'
    json_native = True

    def __init__(self, uri, plugin, flavour):
        self.uri = uri
        self.session = psycopg2.connect(uri)
        self.flavour = flavour

    def get_table(self, identifier: str):
        raise NotImplementedError

    def reconnect(self):
        """Reconnect to the data backend."""
        self.session = psycopg2.connect(self.uri)

    def _run_query(self, query, commit: bool = False):
        start = time.time()
        logging.info(f"Executing query: {query}")
        query = textwrap.dedent(query)
        try:
            with self.session.cursor() as cur:
                cur.execute(query)
                try:
                    result = cur.fetchall()
                except psycopg2.ProgrammingError as e:
                    if 'no results to fetch' in str(e):
                        result = []
                    else:
                        raise e
            logging.info(f"Executing query... DONE ({time.time() - start:.2f}s)")
            if commit:
                self.session.commit()
        except Exception as e:
            self.session.rollback()
            logging.error(f"Error executing query {query}: {e}")
            logging.error(traceback.format_exc())
            raise e
        return result

    def drop_table(self, table: str):
        """Drop data from table.

        :param table: The table to drop.
        """
        return self._run_query(f'DROP TABLE IF EXISTS "{table}"', commit=True)

    @property
    def db(self):
        """Return the datalayer."""
        return self._db

    @db.setter
    def db(self, value):
        """Set the datalayer.

        :param value: The datalayer.
        """
        self._db = value

    def create_tables_and_schemas(self, events: t.List[CreateTable]):
        """Create tables and schemas in the data-backend.

        :param events: The events to create.
        """
        for ev in events:
            self.create_table_and_schema(
                ev.identifier, Schema.build(**ev.fields), ev.primary_id
            )

    def create_table_and_schema(self, identifier: str, schema: Schema, primary_id: str):
        """Create a schema in the data-backend.

        :param identifier: The identifier of the schema.
        :param schema: The schema to create.
        :param primary_id: The primary id of the schema.
        """
        if identifier in self.list_tables():
            return
        native_schema = superduper_to_postgres_schema(schema, primary_id)
        native_schema_str = ',\n    '.join(
            f'"{k}" {v}' for k, v in native_schema.items() 
        )
        q = CREATE_TABLE.format(
            identifier=identifier, primary_id=primary_id, schema=native_schema_str
        )
        self._run_query(q)

    def check_output_dest(self, predict_id) -> bool:
        """Check if the output destination exists.

        :param predict_id: The identifier of the output destination.
        """
        return CFG.output_prefix + predict_id in self.list_tables()

    def _merge_schemas(self, tables: str):
        """Merge schemas.

        :param tables: The tables to merge.
        """
        fields = {}
        for tab in tables:
            tab = self.get_table(tab)
            fields.update(
                {
                    f.name.removeprefix('"').removesuffix('"'): f.datatype
                    for f in tab.schema.fields
                    if f.name not in fields
                }
            )
        return fields

    def drop(self, force: bool = False):
        """Drop the databackend.

        :param force: If ``True``, don't ask for confirmation.
        """
        if not force and not click.confirm(
            "Are you sure you want to drop the database?", default=False
        ):
            return
        for table in self.list_tables():
            logging.info(f"Dropping table {table}")
            self.drop_table(table)
            logging.info(f"Dropping table {table}... DONE")

    def list_tables(self):
        """List all tables or collections in the database."""
        sql = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
              AND table_type IN ('BASE TABLE', 'VIEW')
            ORDER BY table_name;
        """
        with self.session.cursor() as cur:
            cur.execute(sql)
            results = cur.fetchall()
        return [r[0] for r in results]

    ########################################################
    # Abstract methods/ optional methods to be implemented #
    ########################################################

    def random_id(self):
        """Generate a random id."""
        return str(uuid.uuid4().hex)[:16]

    def _fill_primary_id(self, raw_documents, primary_id):
        ids = []
        for r in raw_documents:
            if primary_id not in r:
                r[primary_id] = self.random_id()
            ids.append(r[primary_id])
        return ids

    def _get_columns(self, table_name: str):
        """Get the columns of a table.
        
        :param table_name: The name of the table.
        """
        return [r[0] for r in self._run_query(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = '{table_name}';
        """)]

    def _handle_json_columns(self, raw_documents, table_name):
        s = self.db.metadata.get_schema(table_name)
        json_fields = [
            f for f in s.fields if getattr(s[f], 'dtype', None) == 'json'
        ]
        for f in json_fields:
            for r in raw_documents:
                if f not in r:
                    continue
                r[f] = json.dumps(r[f])

    def insert(self, table_name, raw_documents, primary_id: str | None = None):
        """Insert data into the database.

        :param table: The table to insert into.
        :param raw_documents: The (encoded) documents to insert.
        """
        if primary_id is None:
            primary_id = self.db.metadata.get_primary_id(table_name)

        if len(raw_documents) == 0:
            return []

        ids = self._fill_primary_id(raw_documents, primary_id)
        cols = self._get_columns(table_name)
        def get_row(row):
            return [row[col] if col in row else None for col in cols]
        self._handle_json_columns(raw_documents, table_name)
        values = [get_row(r) for r in raw_documents]
        cols = ', '.join(f'"{c}"' for c in cols)
        with self.session.cursor() as cur:
            sql = INSERT.format(table=table_name, columns=cols)
            try:
                execute_values(
                    cur, sql, values
                )
            except Exception as e:
                logging.error(f"Error executing query: {e}")
                logging.error(traceback.format_exc())
                self.session.rollback()
                raise e
        self.session.commit()
        return ids

    def replace(self, table: str, condition: t.Dict, r: t.Dict) -> t.List[str]:
        """Replace data.

        :param table: The table to insert into.
        :param condition: The condition to update.
        :param r: The document to replace.
        """


        sql = """
        INSERT INTO "{table}" ({primary_id}, {columns})
        VALUES %s
        ON CONFLICT ({primary_id})
        DO UPDATE SET
            {excluded_columns}
        """

        cols = self.db.metadata.get_columns(table)
        primary_id = self.db.metadata.get_primary_id(table)
        cols = [c for c in cols if c != primary_id]
        excluded_columns = '\n'.join([
            f'  {c} = EXCLUDED.{c}'
            for c in cols
        ])
        sql = sql.format(
            table=table,
            columns=cols,
            primary_id=primary_id,
            excluded_columns=excluded_columns,
        )

        if condition:
            sql += " WHERE "
            clause = []
            for k, v in condition.items():
                if isinstance(v, str):
                    v = f"'{v}'"
                elif isinstance(v, (list, tuple)):
                    raise NotImplemented
                clause.append(f'"{k}" = {v}')
            sql += " AND ".join(clause)

        try:
            with self.session.cursor() as cur:
                cur.execute(sql, (r[primary_id], *[r[c] for c in cols]))
        except Exception as e:
            logging.error(f"Error executing query: {e}")
            logging.error(traceback.format_exc())
            self.session.rollback()
            raise e

        self.session.commit()

    def update(self, table: str, condition: t.Dict, key: str, value: t.Any):
        """Update data in the database.

        :param table: The table to update.
        :param condition: The condition to update.
        :param key: The key to update.
        :param value: The value to update.
        """
        conditions = []
        for k, v in condition.items():
            if isinstance(v, str):
                v = f"'{v}'"
            elif isinstance(v, (list, tuple)):
                raise NotImplementedError("List or tuple conditions are not supported.")
            conditions.append(f'"{k}" = {v}')

        conditions = " AND ".join(conditions)
        sql = f"UPDATE \"{table}\" SET {key} = %s WHERE {conditions}"
        schema = self.db.metadata.get_schema(table)
        with self.session.cursor() as cur:
            logging.info(f"Executing query: {sql} with value: {value}")
            if isinstance(value, str):
                value = f"'{value}'"
            if getattr(schema[key], 'dtype', None) == 'json':
                value = json.dumps(value)
            try:
                cur.execute(sql, (value,))
            except Exception as e:
                logging.error(f"Error executing query: {e}")
                logging.error(traceback.format_exc())
                self.session.rollback()
                raise e
        self.session.commit()

    def delete(self, table: str, condition: t.Dict):
        """Update data in the database.

        :param table: The table to update.
        :param condition: The condition to update.
        """
        terms = []
        for k, v in condition.items():
            if isinstance(v, str):
                v = f"'{v}'"
            terms.append(f'"{k}" = {v}')
        condition = " AND ".join(terms)
        q = f'DELETE FROM "{table}" WHERE {condition}'
        logging.info(f"Executing query: {q}")
        self._run_query(f'DELETE FROM "{table}" WHERE {condition}')

    def missing_outputs(self, query, predict_id):
        """Get missing outputs.

        :param query: The query to get the missing outputs of.
        :param predict_id: The identifier of the output destination.
        """
        pid = self.primary_id(query.table)

        sql = f"""
        SELECT {query.table}.{pid} 
        FROM {query.table}
        LEFT JOIN "{CFG.output_prefix}{predict_id}" ON {query.table}.{pid} = {CFG.output_prefix}{predict_id}._source
        WHERE {CFG.output_prefix}{predict_id}._source IS NULL
        """

        filter = query.decomposition.filter
        if filter:
            raise NotImplementedError

        return [r[0] for r in self._run_query(sql)]

    def _build_schema(self, query: Query):
        """Build the schema of a query.

        :param query: The query to build the schema of.
        """
        return self.get_table(query.table).schema

    def select(self, query: Query, primary_id: str | None = None) -> t.List[t.Dict]:
        """Select data from the database.

        :param query: The query to perform.
        """
        q = map_superduper_query_to_postgres_query(query)
        cols = list(query.decomposition.columns)
        for i, c in enumerate(cols):
            if isinstance(c, Query):
                cols[i] = c.execute()
        try:
            results = self._run_query(q)
        except Exception as e:
            print(traceback.format_exc())
            import pdb; pdb.set_trace()
        results = [{col: v for col, v in zip(cols, result)} for result in results]
        return results

    def execute_native(self, query: str):
        """Execute a native query.

        :param query: The query to execute.
        """
        results = self._run_query(query)
        out = []
        for r in results:
            out.append(r.as_dict())
        return out

