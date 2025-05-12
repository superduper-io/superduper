import json
import os
import threading
import time
import typing as t
import uuid

import click
import numpy as np
import pandas
from snowflake.snowpark.functions import col
from snowflake.snowpark.types import BooleanType, StringType, VariantType
from superduper import CFG, logging
from superduper.backends.base.data_backend import BaseDataBackend
from superduper.base.query import Query
from superduper.base.schema import Schema
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from superduper_snowflake.connect import connect
from superduper_snowflake.query import map_superduper_query_to_snowpark_query
from superduper_snowflake.schema import superduper_to_snowflake_schema

# Hybrid tables are a feature of Snowflake which are a proxy
# for transactional tables.
create_table = """
CREATE OR REPLACE TABLE "{identifier}" (
    {schema}
);
"""

insert_to_row = """
INSERT INTO "{table}" ({columns}) 
  SELECT
    {values}
"""


db_lock = threading.Lock()
SESSION_DIR = os.environ.get('SNOWFLAKE_SESSION_DIR') or '/snowflake/session'


class _SnowflakeTokenWatcher(FileSystemEventHandler):
    timeout = 60

    def __init__(self, databackend):
        super().__init__()
        self.databackend = databackend

    def on_any_event(self, event):
        logging.warn(str(event))

        if event.src_path.endswith('data_tmp') and event.event_type == 'moved':
            with db_lock:
                self.databackend.reconnect()


def _watch_token_file(databackend):
    observer = Observer()
    handler = _SnowflakeTokenWatcher(databackend)

    logging.info(f'Starting Snowflake token watcher on {SESSION_DIR}/token')

    observer.schedule(handler, SESSION_DIR, recursive=False)
    observer.start()
    logging.info('Started Snowflake token watcher')
    return observer


class SnowflakeDataBackend(BaseDataBackend):
    """Snowflake data backend."""

    def __init__(self, uri, plugin, flavour):
        self.uri = uri
        self.session, self.schema = connect(uri)
        self.observer = None
        if self.uri == 'snowflake://':
            self.observer = _watch_token_file(self)

    def reconnect(self):
        """Reconnect to the data backend."""
        self.session = connect(self.uri)

    def _run_query(self, query):
        start = time.time()
        logging.info(f"Executing query: {query}")
        result = self.session.sql(query).collect()
        logging.info(f"Executing query... DONE ({time.time() - start:.2f}s)")
        return result

    def _run_bind_query(self, sql, params):
        start = time.time()
        logging.info(f"Executing query: {sql} with params: {params}")
        result = self.session.sql(sql).bind(params).collect()
        logging.info(f"Executing query... DONE ({time.time() - start:.2f}s)")
        return result

    def drop_table(self, table: str):
        """Drop data from table.

        :param table: The table to drop.
        """
        self._run_query(f'DROP TABLE IF EXISTS {self.schema}."{table}"')

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

    def create_table_and_schema(self, identifier: str, schema: Schema, primary_id: str):
        """Create a schema in the data-backend.

        :param identifier: The identifier of the schema.
        :param schema: The schema to create.
        :param primary_id: The primary id of the schema.
        """
        if identifier in self.list_tables():
            return
        native_schema_str = ',\n    '.join(
            superduper_to_snowflake_schema(schema, primary_id)
        )
        q = create_table.format(
            identifier=identifier, primary_id=primary_id, schema=native_schema_str
        )
        return self._run_query(q)

    def check_output_dest(self, predict_id) -> bool:
        """Check if the output destination exists.

        :param predict_id: The identifier of the output destination.
        """
        return CFG.output_prefix + predict_id in self.list_tables()

    def get_table(self, identifier):
        """Get a table or collection from the database.

        :param identifier: The identifier of the table or collection.
        """
        return self.session.table(f'"{identifier}"')

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
        results = self.session.sql("SHOW TABLES").collect()
        return [r.name for r in results]

    ########################################################
    # Abstract methods/ optional methods to be implemented #
    ########################################################

    def random_id(self):
        """Generate a random id."""
        return str(uuid.uuid4().hex)

    def _fill_primary_id(self, raw_documents, primary_id):
        ids = []
        for r in raw_documents:
            if primary_id not in r:
                r[primary_id] = self.random_id()
            ids.append(r[primary_id])
        return ids

    @staticmethod
    def _docs_to_dataframe(raw_documents, cols):
        df = pandas.DataFrame(raw_documents)
        df = df.to_dict(orient='records')

        def get_row(row):
            return [row[col] if col in row else None for col in cols]

        return list(map(get_row, df))

    def insert(self, table_name, raw_documents, primary_id: str | None = None):
        """Insert data into the database.

        :param table: The table to insert into.
        :param raw_documents: The (encoded) documents to insert.
        """
        if primary_id is None:
            primary_id = self.db.load('Table', table_name).primary_id

        if len(raw_documents) == 0:
            return []
        # if len(raw_documents) == 1:
        #     return [self._insert_row(table_name, raw_documents[0], primary_id)]

        schema = self.get_table(table_name).schema
        ids = self._fill_primary_id(raw_documents, primary_id)

        # columns are quoted due to case insensitive nature of snowflake
        cols = [c.name.removeprefix('"').removesuffix('"') for c in schema.fields]
        rows = self._docs_to_dataframe(raw_documents, cols)
        native_df = self.session.create_dataframe(rows, schema=schema)
        native_df.write.saveAsTable(f'"{table_name}"', mode='append')
        return ids

    def _insert_row(self, table: str, r: t.Dict, primary_id: str):
        t = self.session.table(f'"{table}"')
        if primary_id not in r:
            id = self._fill_primary_id([r], primary_id)[0]
        else:
            id = r[primary_id]
            del r[primary_id]
        to_insert = []
        columns = [f'"{primary_id}"']
        to_insert = [f"'{id}'"]
        for f in t.schema.fields:
            hr = f.name.removeprefix('"').removesuffix('"')
            if f.name == f'"{primary_id}"':
                continue
            columns.append(f.name)
            if hr not in r:
                to_insert.append("NULL")
            elif r[hr] is None:
                to_insert.append("NULL")
            elif f.datatype == VariantType():
                value = r[hr]
                if isinstance(value, str):
                    value = f"PARSE_JSON('{value}')"
                else:
                    value = f"PARSE_JSON('{json.dumps(value)}')"
                value = value.replace("'", "''")
                to_insert.append(value)
            elif f.datatype == BooleanType():
                to_insert.append(f"{str(r[hr]).upper()}")
            elif f.datatype == StringType():
                to_insert.append(f"'{r[hr]}'")
            else:
                to_insert.append(f'{r[hr]}')
        columns = ', '.join(columns)
        to_insert = ', '.join(to_insert)
        statement = insert_to_row.format(table=table, columns=columns, values=to_insert)
        self._run_query(statement)
        return id

    def replace(self, table: str, condition: t.Dict, r: t.Dict) -> t.List[str]:
        """Replace data.

        :param table: The table to insert into.
        :param condition: The condition to update.
        :param r: The document to replace.
        """
        t = self.get_table(table)
        cond = None
        for k, v in condition.items():
            if isinstance(v, str):
                v = str(v)
            k = quote_identifier(k)
            expr = t[k] == v
            cond = expr if cond is None else (cond & expr)

        update_r = {}
        for k, v in r.items():
            k = quote_identifier(k)
            if isinstance(v, str):
                v = str(v)
            update_r[k] = v

        t.update(update_r, cond)
        return list(r.keys())

    def update(self, table: str, condition: t.Dict, key: str, value: t.Any):
        """Update data in the database.

        :param table: The table to update.
        :param condition: The condition to update.
        :param key: The key to update.
        :param value: The value to update.
        """
        return self.replace(table, condition, {key: value})

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
        df = map_superduper_query_to_snowpark_query(self.session, query, pid)
        output_df = self.session.table(f'"{CFG.output_prefix + predict_id}"')
        columns = output_df.columns
        columns = [c for c in columns if c != '"id"']
        output_df = output_df.select(*columns)
        output_df = output_df.with_column_renamed('"_source"', '"_source_target"')

        joined_df = df.join(
            output_df, df[f'"{pid}"'] == output_df['"_source_target"'], join_type="left"
        )
        anti_join = joined_df.filter(col('"_source_target"').is_null())

        return (
            anti_join.select(f'"{pid}"')
            .to_pandas()[pid.removeprefix('"').removesuffix('"')]
            .tolist()
        )

    def _build_schema(self, query: Query):
        """Build the schema of a query.

        :param query: The query to build the schema of.
        """
        return self.get_table(query.table).schema

    def select(self, query: Query, primary_id: str | None = None) -> t.List[t.Dict]:
        """Select data from the database.

        :param query: The query to perform.
        """
        q = map_superduper_query_to_snowpark_query(
            self.session,
            query,
            primary_id or self.primary_id(query.table),
        )
        start = time.time()
        logging.info(f"Executing query: {query}")
        result = q.to_pandas()
        result = result.replace({np.nan: None})
        logging.info(f"Executing query... DONE ({time.time() - start:.2f}s)")
        merged_schemas = self._merge_schemas(query.tables)
        for k in result.columns:
            if merged_schemas.get(k) == VariantType():
                result[k] = result[k].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
        return result.to_dict(orient='records')

    def execute_native(self, query: str):
        """Execute a native query.

        :param query: The query to execute.
        """
        results = self._run_query(query)
        out = []
        for r in results:
            out.append(r.as_dict())
        return out


def quote_identifier(identifier: str) -> str:
    """Snowflake-safe identifier quoting."""
    return '"' + identifier.replace('"', '""') + '"'
