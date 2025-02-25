import json
import uuid
import typing as t

import click
import pandas

from snowflake.snowpark.types import VariantType
from snowflake.snowpark.functions import col

from superduper_snowflake.query import map_superduper_query_to_snowpark_query
from superduper.base.query import Query
from superduper.base.schema import Schema
from superduper_snowflake.connect import connect
from superduper import logging

from superduper.backends.base.data_backend import BaseDataBackend
from superduper import CFG

from superduper_snowflake.schema import superduper_to_snowflake_schema


# Hybrid tables are a feature of Snowflake which are a proxy
# for transactional tables.
create_table = """
CREATE OR REPLACE HYBRID TABLE "{identifier}" (
    {schema}
);
"""


class SnowflakeDataBackend(BaseDataBackend):
    batched: bool = False

    def __init__(self, uri, plugin, flavour):
        self.uri = uri
        self.session, self.schema = connect(uri)

    def reconnect(self):
        self.session = connect(self.uri)

    def drop_table(self, table: str):
        """Drop data from table.

        :param table: The table to drop.
        """
        self.session.sql(f'DROP TABLE IF EXISTS {self.schema}."{table}"').collect()

    def random_id(self):
        """Generate random-id."""
        return str(uuid.uuid4().hex)

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

    def create_table_and_schema(
        self, identifier: str, schema: Schema, primary_id: str
    ):
        """Create a schema in the data-backend.

        :param identifier: The identifier of the schema.
        :param schema: The schema to create.
        :param primary_id: The primary id of the schema.
        """
        if identifier in self.list_tables():
            return
        native_schema_str = ',\n    '.join(superduper_to_snowflake_schema(schema, primary_id))
        q = create_table.format(identifier=identifier, primary_id=primary_id, schema=native_schema_str)
        logging.info(f"Creating table with query: {q}")
        self.execute_native(q)
        logging.info('Creating table ... DONE')

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
        for t in tables:
            t = self.get_table(t)
            fields.update({f.name.removeprefix('"').removesuffix('"'): f.datatype for f in t.schema.fields if f.name not in fields})
        return fields

    def drop(self, force: bool = False):
        """Drop the databackend.

        :param force: If ``True``, don't ask for confirmation.
        """
        if not force and not click.confirm(
            "Are you sure you want to drop the database?", default=False
        ):
            return
        for t in self.list_tables():
            logging.info(f"Dropping table {t}")
            self.drop_table(t)
            logging.info(f"Dropping table {t}... DONE")

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
        for i, r in enumerate(raw_documents):
            if primary_id not in r:
                r[primary_id] = self.random_id()
            ids.append(r[primary_id])
        return ids

    @staticmethod
    def _docs_to_dataframe(raw_documents, cols):
        df = pandas.DataFrame(raw_documents)
        df = df.to_dict(orient='records')
        get_row = lambda row: [row[col] if col in row else None for col in cols]
        return list(map(get_row, df))

    def insert(self, table_name, raw_documents, primary_id: str | None = None):
        """Insert data into the database.

        :param table: The table to insert into.
        :param raw_documents: The (encoded) documents to insert.
        """

        schema = self.get_table(table_name).schema
        if primary_id is None:
            primary_id = self.db.load('Table', table_name).primary_id
        ids = self._fill_primary_id(raw_documents, primary_id)
        # columns are quoted due to case insensitive nature of snowflake
        cols = [c.name.removeprefix('"').removesuffix('"') for c in schema.fields]
        rows = self._docs_to_dataframe(raw_documents, cols)
        native_df = self.session.create_dataframe(rows, schema=schema)
        native_df.write.saveAsTable(f'"{table_name}"', mode='append')
        return ids

    def replace(self, table: str, condition: t.Dict, r: t.Dict) -> t.List[str]:
        """Replace data.

        :param table: The table to insert into.
        :param condition: The condition to update.
        :param r: The document to replace.
        """
        repl = r.copy()
        for k, v in r.items():
            if isinstance(v, str):
                repl[k] = f"'{v}'"
        repl_part = ", ".join([f"{k} = {v}" for k, v in repl.items()])
        self.session.sql(
            f"UPDATE {table} SET {repl_part} WHERE {condition}"
        )

    def update(self, table: str, condition: t.Dict, key: str, value: t.Any):
        """Update data in the database.

        :param table: The table to update.
        :param condition: The condition to update.
        :param key: The key to update.
        :param value: The value to update.
        """
        terms = []
        for k, v in condition.items():
            if isinstance(v, str):
                v = f"'{v}'"
            terms.append(f'"{k}" = {v}')
        condition = " AND ".join(terms)
        if isinstance(value, str):
            value = f"'{value}'"
        self.session.sql(
            f'UPDATE "{table}" SET "{key}" = {value} WHERE {condition}'
        ).collect()

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
        self.session.sql(
            f'DELETE FROM "{table}" WHERE {condition}'
        ).collect()
        logging.info(f"Executing query... DONE")

    def missing_outputs(self, query, predict_id):
        pid = self.primary_id(query)
        df = map_superduper_query_to_snowpark_query(self.session, query, pid)
        output_df = self.session.table(f'"{CFG.output_prefix + predict_id}"')
        columns = output_df.columns
        columns = [c for c in columns if c != '"id"']
        output_df = output_df.select(*columns)

        joined_df = df.join(output_df, df[f'"{pid}"'] == output_df['"_source"'], join_type="left")
        anti_join = joined_df.filter(col(f'"_source"').is_null())

        return anti_join.select(f'"{pid}"').to_pandas()[pid.removeprefix('"').removesuffix('"')].tolist()

    def primary_id(self, query: Query) -> str:
        """Get the primary id of a query.

        :param query: The query to get the primary id of.
        """
        return self.get_table(query.table).schema[0].name.removeprefix('"').removesuffix('"')

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
            primary_id or self.primary_id(query),
        )
        result = q.to_pandas()
        merged_schemas = self._merge_schemas(query.tables)
        for k in result.columns:
            if merged_schemas[k] == VariantType():
                result[k] = result[k].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        return result.to_dict(orient='records')

    def execute_native(self, query: str):
        """Execute a native query.

        :param query: The query to execute.
        """
        results = self.session.sql(query).collect()
        out = []
        for r in results:
            out.append(r.as_dict())
        return out

    def execute_events(self, events):
        ...
        return super().execute_events(events)