import os
import typing as t
from warnings import warn

import ibis
import pandas
from ibis.backends.base import BaseBackend
from pandas.core.frame import DataFrame
from sqlalchemy.exc import NoSuchTableError

from superduperdb import logging
from superduperdb.backends.base.data_backend import BaseDataBackend
from superduperdb.backends.ibis.db_helper import get_db_helper
from superduperdb.backends.ibis.field_types import FieldType, dtype
from superduperdb.backends.ibis.query import IbisQuery
from superduperdb.backends.ibis.utils import convert_schema_to_fields
from superduperdb.backends.local.artifacts import FileSystemArtifactStore
from superduperdb.backends.sqlalchemy.metadata import SQLAlchemyMetadata
from superduperdb.base.enums import DBType
from superduperdb.components.datatype import DataType
from superduperdb.components.schema import Schema
from superduperdb.components.table import Table

BASE64_PREFIX = 'base64:'
INPUT_KEY = '_source'


class IbisDataBackend(BaseDataBackend):
    """Ibis data backend for the database.

    :param conn: The connection to the database.
    :param name: The name of the database.
    :param in_memory: Whether to store the data in memory.
    """

    db_type = DBType.SQL

    def __init__(self, conn: BaseBackend, name: str, in_memory: bool = False):
        super().__init__(conn=conn, name=name)
        self.in_memory = in_memory
        self.dialect = getattr(conn, 'name', 'base')
        self.db_helper = get_db_helper(self.dialect)

    def get_query_builder(self, table_name):
        """Get the query builder for the data backend.

        :param table_name: Which table to get the query builder for
        """
        return IbisQuery(identifier=table_name, db=self.datalayer)

    def url(self):
        """Get the URL of the database."""
        return self.conn.con.url + self.name

    def build_artifact_store(self):
        """Build artifact store for the database."""
        return FileSystemArtifactStore(conn='.superduperdb/artifacts/', name='ibis')

    def build_metadata(self):
        """Build metadata for the database."""
        return SQLAlchemyMetadata(conn=self.conn.con, name='ibis')

    def insert(self, table_name, raw_documents):
        """Insert data into the database.

        :param table_name: The name of the table.
        :param raw_documents: The data to insert.
        """
        for doc in raw_documents:
            for k, v in doc.items():
                doc[k] = self.db_helper.convert_data_format(v)
        table_name, raw_documents = self.db_helper.process_before_insert(
            table_name, raw_documents
        )
        if not self.in_memory:
            self.conn.insert(table_name, raw_documents)
        else:
            # CAUTION: The following is only tested with pandas.
            if table_name in self.conn.tables:
                t = self.conn.tables[table_name]
                df = pandas.concat([t.to_pandas(), raw_documents])
                self.conn.create_table(table_name, df, overwrite=True)
            else:
                df = pandas.DataFrame(raw_documents)
                self.conn.create_table(table_name, df)

            if self.conn.backend_table_type == DataFrame:
                df.to_csv(os.path.join(self.name, table_name + '.csv'), index=False)

    def create_output_dest(
        self,
        predict_id: str,
        datatype: t.Union[FieldType, DataType],
        flatten: bool = False,
    ):
        """Create a table for the output of the model.

        :param predict_id: The identifier of the prediction.
        :param datatype: The data type of the output.
        :param flatten: Whether to flatten the output.
        """
        # TODO: Support output schema
        msg = (
            "Model must have an encoder to create with the"
            f" {type(self).__name__} backend."
        )
        assert datatype is not None, msg
        if isinstance(datatype, FieldType):
            output_type = dtype(datatype.identifier)
        else:
            output_type = datatype

        if flatten:
            fields = {
                INPUT_KEY: dtype('string'),
                'id': dtype('string'),
                'output': output_type,
            }
            return Table(
                primary_id='id',
                identifier=f'_outputs.{predict_id}',
                schema=Schema(identifier=f'_schema/{predict_id}', fields=fields),
            )
        else:
            fields = {
                INPUT_KEY: dtype('string'),
                'output': output_type,
                'id': dtype('string'),
            }
            return Table(
                identifier=f'_outputs.{predict_id}',
                schema=Schema(identifier=f'_schema/{predict_id}', fields=fields),
            )

    def check_output_dest(self, predict_id) -> bool:
        """Check if the output destination exists.

        :param predict_id: The identifier of the prediction.
        """
        try:
            self.conn.table(f'_outputs.{predict_id}')
            return True
        except NoSuchTableError:
            return False

    def create_table_and_schema(self, identifier: str, schema: Schema):
        """Create a schema in the data-backend.

        :param identifier: The identifier of the table.
        :param mapping: The mapping of the schema.
        """
        mapping = convert_schema_to_fields(schema)
        if 'id' not in mapping:
            mapping['id'] = 'string'
        try:
            mapping = self.db_helper.process_schema_types(mapping)
            t = self.conn.create_table(identifier, schema=ibis.schema(mapping))
        except Exception as e:
            if 'exists' in str(e) or 'override' in str(e):
                warn("Table already exists, skipping...")
                t = self.conn.table(identifier)
            else:
                raise e
        return t

    def drop(self, force: bool = False):
        """Drop tables or collections in the database.

        :param force: Whether to force the drop.
        """
        raise NotImplementedError(
            "Dropping tables needs to be done in each DB natively"
        )

    def get_table_or_collection(self, identifier):
        """Get a table or collection from the database.

        :param identifier: The identifier of the table or collection.
        """
        return self.conn.table(identifier)

    def disconnect(self):
        """Disconnect the client."""

        # TODO: implement me

    def list_tables_or_collections(self):
        """List all tables or collections in the database."""
        return self.conn.list_tables()

    @staticmethod
    def infer_schema(data: t.Mapping[str, t.Any], identifier: t.Optional[str] = None):
        """Infer a schema from a given data object.

        :param data: The data object
        :param identifier: The identifier for the schema, if None, it will be generated
        :return: The inferred schema
        """
        from superduperdb.misc.auto_schema import infer_schema

        return infer_schema(data, identifier=identifier, ibis=True)

    def auto_create_table_schema(self, db, table_name, documents):
        """Auto create table schema.

        For Ibis, we need to create the table schema before inserting the data.
        The function will infer the schema from the first document and create the table
        if the table does not exist.

        :param db: The datalayer instanace
        :param table_name: The table name
        :param documents: The documents
        """
        try:
            table = db.tables[table_name]
            return table
        except FileNotFoundError:
            logging.info(f"Table {table_name} does not exist, auto creating...")
        # Should we need to check all the documents?
        document = documents[0]
        schema = document.schema or self.infer_schema(document)
        table = Table(identifier=table_name, schema=schema)
        if table.primary_id not in schema.fields:
            table.schema.fields[table.primary_id] = dtype('str')
        logging.info(f"Creating table {table_name} with schema {schema.fields_set}")
        db.apply(table)
