import glob
import os
import typing as t
from warnings import warn

import ibis
import pandas
from pandas.core.frame import DataFrame
from sqlalchemy.exc import NoSuchTableError

from superduper import logging
from superduper.backends.base.data_backend import BaseDataBackend
from superduper.backends.base.metadata import MetaDataStoreProxy
from superduper.backends.ibis.db_helper import get_db_helper
from superduper.backends.ibis.field_types import FieldType, dtype
from superduper.backends.ibis.query import IbisQuery
from superduper.backends.ibis.utils import convert_schema_to_fields
from superduper.backends.local.artifacts import FileSystemArtifactStore
from superduper.backends.sqlalchemy.metadata import SQLAlchemyMetadata
from superduper.base.enums import DBType
from superduper.components.datatype import DataType
from superduper.components.schema import Schema
from superduper.components.table import Table

BASE64_PREFIX = 'base64:'
INPUT_KEY = '_source'


def _connection_callback(uri, flavour):
    if flavour == 'pandas':
        uri = uri.split('://')[-1]
        csv_files = glob.glob(uri)
        dir_name = os.path.dirname(uri)
        tables = {}
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            if os.path.getsize(csv_file) == 0:
                df = pandas.DataFrame()
            else:
                df = pandas.read_csv(csv_file)
            tables[filename.split('.')[0]] = df
        ibis_conn = ibis.pandas.connect(tables)
        in_memory = True
        return ibis_conn, dir_name, in_memory
    else:
        name = uri.split('//')[0]
        in_memory = False
        ibis_conn = ibis.connect(uri)
        return ibis_conn, name, in_memory


class IbisDataBackend(BaseDataBackend):
    """Ibis data backend for the database.

    :param uri: URI to the databackend database.
    :param flavour: Flavour of the databackend.
    """

    db_type = DBType.SQL

    def __init__(self, uri: str, flavour: t.Optional[str] = None):
        self.connection_callback = lambda: _connection_callback(uri, flavour)
        conn, name, in_memory = self.connection_callback()
        super().__init__(uri=uri, flavour=flavour)
        self.conn = conn
        self.name = name
        self.in_memory = in_memory
        self._setup(conn)

    def _setup(self, conn):
        self.dialect = getattr(conn, 'name', 'base')
        self.db_helper = get_db_helper(self.dialect)

    def reconnect(self):
        """Reconnect to the database client."""
        # Reconnect to database.
        conn, _, _ = self.connection_callback()
        self.conn = conn
        self._setup(conn)

    def get_query_builder(self, table_name):
        """Get the query builder for the data backend.

        :param table_name: Which table to get the query builder for
        """
        return IbisQuery(table=table_name, db=self.datalayer)

    def url(self):
        """Get the URL of the database."""
        return self.conn.con.url + self.name

    def build_artifact_store(self):
        """Build artifact store for the database."""
        return FileSystemArtifactStore(conn='.superduper/artifacts/', name='ibis')

    def build_metadata(self):
        """Build metadata for the database."""

        def callback():
            return self.conn.con, self.name

        return MetaDataStoreProxy(SQLAlchemyMetadata(callback=callback))

    def insert(self, table_name, raw_documents):
        """Insert data into the database.

        :param table_name: The name of the table.
        :param raw_documents: The data to insert.
        """
        for doc in raw_documents:
            for k, v in doc.items():
                doc[k] = self.db_helper.convert_data_format(v)
        table_name, raw_documents = self.db_helper.process_before_insert(
            table_name,
            raw_documents,
            self.conn,
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

    def drop_outputs(self):
        """Drop the outputs."""
        raise NotImplementedError

    def drop_table_or_collection(self, name: str):
        """Drop the table or collection.

        Please use with caution as you will lose all data.
        :param name: Table name to drop.
        """
        return self.db.databackend.conn.drop_table(name)

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
        except (NoSuchTableError, ibis.IbisError):
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
        from superduper.misc.auto_schema import infer_schema

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
