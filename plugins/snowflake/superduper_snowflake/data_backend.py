import uuid
import typing as t

import click
import pandas

from superduper_snowflake.schema import (
    superduper_fields_to_snowpark_schema,
    snowpark_cols_to_schema,
)
from superduper.base.query import Query
from superduper.base.schema import Schema
from superduper_snowflake.connect import connect
from superduper import logging

from superduper.backends.base.data_backend import BaseDataBackend
from superduper import CFG


class SnowflakeDataBackend(BaseDataBackend):
    def __init__(self, uri, plugin, flavour):
        self.uri = uri
        self.session = self.connect(self.uri)

    def reconnect(self):
        self.snowpark = connect(self.uri)

    def drop_table(self, table: str):
        """Drop data from table.

        :param table: The table to drop.
        """
        self.session.sql(f'DROP TABLE IF EXISTS "{table}"')

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
        logging.info('Skipping create_table_and_schema until insertion')

    def check_output_dest(self, predict_id) -> bool:
        """Check if the output destination exists.

        :param predict_id: The identifier of the output destination.
        """
        return CFG.output_prefix + predict_id in self.list_tables()

    def get_table(self, identifier):
        """Get a table or collection from the database.

        :param identifier: The identifier of the table or collection.
        """
        raise NotImplementedError

    def _drop_table(self, table):
        self.session.sql(f"DROP TABLE IF EXISTS {table}")

    def drop(self, force: bool = False):
        """Drop the databackend.

        :param force: If ``True``, don't ask for confirmation.
        """
        if not click.confirm(
            "Are you sure you want to drop the database?", default=False
        ):
            return
        for t in self._show_tables():
            self.drop_table(t)

    def list_tables(self):
        """List all tables or collections in the database."""
        return list(self.session.sql("SHOW TABLES").collect())

    def reconnect(self):
        """Reconnect to the databackend."""
        self.snowpark = connect(self.uri)

    ########################################################
    # Abstract methods/ optional methods to be implemented #
    ########################################################

    def insert(self, table_name, raw_documents):
        """Insert data into the database.

        :param table: The table to insert into.
        :param documents: The documents to insert.
        """
        ibis_schema = self.conn.table(table_name).schema()
        t = self.db.load('Table', table_name)
        snowpark_cols = superduper_fields_to_snowpark_schema(t.fields)
        df = pandas.DataFrame(raw_documents)
        rows = list(df.itertuples(index=False, name=None))
        columns = list(ibis_schema.keys())
        df = df.to_dict(orient='records')
        get_row = lambda row: [row[col] for col in columns]
        rows = list(map(get_row, df))
        snowpark_schema = snowpark_cols_to_schema(snowpark_cols, columns)
        native_df = self.snowpark.create_dataframe(rows, schema=snowpark_schema)
        return native_df.write.saveAsTable(f'"{table_name}"', mode='append')

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
        condition = condition[:]
        for i, c in enumerate(condition):
            if isinstance(c, str):
                condition[i] = f"'{c}'"
        condition = " AND ".join([f"{k} = {v}" for k, v in condition.items()])
        if isinstance(value, str):
            value = f"'{value}'"
        self.session.sql(
            f"UPDATE {table} SET {key} = {value} WHERE {condition}"
        )

    def delete(self, table: str, condition: t.Dict):
        """Update data in the database.

        :param table: The table to update.
        :param condition: The condition to update.
        """
        condition = condition[:]
        for i, c in enumerate(condition):
            if isinstance(c, str):
                condition[i] = f"'{c}'"
        condition = " AND ".join([f"{k} = {v}" for k, v in condition.items()])
        self.session.sql(
            f"DELETE FROM {table} WHERE {condition}"
        )

    def missing_outputs(self, query: Query, predict_id: str) -> t.List[str]:
        """Get missing outputs from an outputs query.

        This method will be used to perform an anti-join between
        the input and the outputs table, and return the missing ids.

        :param query: The query to perform.
        :param predict_id: The predict id.
        """
        input_df1 = self._create_select_dataframe(query)
        output_df = self.session.createDataFrame(self.conn.table(CFG.output_prefix + predict_id))
        primary_id = ...
        df1 = self.session.
        joined_df = df1.join(df2, df1["KEY_COL"] == df2["KEY_COL"], join_type="left")
        anti_join_df = joined_df.filter(col("TABLE2.KEY_COL").is_null())

        raise NotImplementedError

    def primary_id(self, query: Query) -> str:
        """Get the primary id of a query.

        :param query: The query to get the primary id of.
        """
        raise NotImplementedError

    def select(self, query: Query) -> t.List[t.Dict]:
        """Select data from the database.

        :param query: The query to perform.
        """
        t = self.get_table(query.table)
