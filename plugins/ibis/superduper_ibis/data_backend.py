import glob
import os
import typing as t
from warnings import warn

import click
import ibis
import pandas
from pandas.core.frame import DataFrame
from sqlalchemy.exc import NoSuchTableError
from superduper import CFG, logging
from superduper.backends.base.data_backend import BaseDataBackend
from superduper.backends.base.metadata import MetaDataStoreProxy
from superduper.backends.local.artifacts import FileSystemArtifactStore
from superduper.base import exceptions
from superduper.base.enums import DBType
from superduper.components.datatype import BaseDataType
from superduper.components.schema import Schema
from superduper.components.table import Table

from superduper_ibis.db_helper import get_db_helper
from superduper_ibis.field_types import FieldType, dtype
from superduper_ibis.query import IbisQuery
from superduper_ibis.utils import convert_schema_to_fields

BASE64_PREFIX = "base64:"
INPUT_KEY = "_source"


def _snowflake_connection_callback():
    # In the Snowflake native apps framework, the
    # inbuild database is provided by env variables
    # and authentication is via OAuth with a
    # mounted token. In this case, as a convention
    # we connect with `"snowflake://"`

    logging.info('Using env variables and OAuth to connect!')

    import snowflake.connector

    conn = snowflake.connector.connect(
        host=os.environ['SNOWFLAKE_HOST'],
        port=int(os.environ['SNOWFLAKE_PORT']),
        account=os.environ['SNOWFLAKE_ACCOUNT'],
        authenticator='oauth',
        token=open('/snowflake/session/token').read(),
        warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
        database=os.environ['SNOWFLAKE_DATABASE'],
        schema=os.environ['SUPERDUPER_DATA_SCHEMA'],
    )

    return ibis.snowflake.from_connection(conn)


def _connection_callback(uri, flavour):
    if flavour == "pandas":
        uri = uri.split("://")[-1]
        csv_files = glob.glob(uri)
        dir_name = os.path.dirname(uri)
        tables = {}
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            if os.path.getsize(csv_file) == 0:
                df = pandas.DataFrame()
            else:
                df = pandas.read_csv(csv_file)
            tables[filename.split(".")[0]] = df
        ibis_conn = ibis.pandas.connect(tables)
        in_memory = True
        return ibis_conn, dir_name, in_memory
    elif uri == 'snowflake://':
        return _snowflake_connection_callback(), 'snowflake', False
    else:
        name = uri.split("//")[0]
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
        self.overwrite = False
        self._setup(conn)

        self.datatype_presets = {'vector': 'superduper.ext.numpy.encoder.Array'}

        if uri.startswith('snowflake://'):
            self.bytes_encoding = 'base64'
            self.datatype_presets = {
                'vector': 'superduper.components.datatype.NativeVector'
            }

    def _setup(self, conn):
        self.dialect = getattr(conn, "name", "base")
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
        return FileSystemArtifactStore(conn=".superduper/artifacts/", name="ibis")

    def build_metadata(self):
        """Build metadata for the database."""
        from superduper_sqlalchemy.metadata import SQLAlchemyMetadata

        try:
            return MetaDataStoreProxy(
                SQLAlchemyMetadata(callback=lambda: (self.conn.con, self.name))
            )
        except Exception as e:
            logging.warn(
                f"Unable to connect to the database with self.conn.con: "
                f"{self.conn.con} and self.name: {self.name}. Error: {e}."
            )
            logging.warn(f"Falling back to using the uri: {self.uri}.")
            return MetaDataStoreProxy(SQLAlchemyMetadata(uri=self.uri))

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
                df.to_csv(os.path.join(self.name, table_name + ".csv"), index=False)

    def check_ready_ids(
        self, query: IbisQuery, keys: t.List[str], ids: t.Optional[t.List[t.Any]] = None
    ):
        """Check if all the keys are ready in the ids.

        :param query: The query object.
        :param keys: The keys to check.
        :param ids: The ids to check.
        """
        if ids:
            query = query.filter(query[query.primary_id].isin(ids))
        conditions = []
        for key in keys:
            conditions.append(query[key].notnull())

        # TODO: Hotfix, will be removed by the refactor PR
        try:
            docs = query.filter(*conditions).select(query.primary_id).execute()
        except Exception as e:
            if "Table not found" in str(e) or "Can't find table" in str(e):
                return []
            else:
                raise e
        ready_ids = [doc[query.primary_id] for doc in docs]
        self._log_check_ready_ids_message(ids, ready_ids)
        return ready_ids

    def drop_outputs(self):
        """Drop the outputs."""
        for table in self.conn.list_tables():
            logging.info(f"Dropping table: {table}")
            if CFG.output_prefix in table:
                self.conn.drop_table(table)

    def drop_table_or_collection(self, name: str):
        """Drop the table or collection.

        Please use with caution as you will lose all data.
        :param name: Table name to drop.
        """
        try:
            return self.conn.drop_table(name)
        except Exception as e:
            msg = "Object found is of type 'VIEW'"
            if msg in str(e):
                return self.conn.drop_view(name)
            raise

    def create_output_dest(
        self,
        predict_id: str,
        datatype: t.Union[FieldType, BaseDataType],
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

        fields = {
            INPUT_KEY: "string",
            "_source": "string",
            "id": "string",
            f"{CFG.output_prefix}{predict_id}": output_type,
        }
        return Table(
            identifier=f"{CFG.output_prefix}{predict_id}",
            schema=Schema(identifier=f"_schema/{predict_id}", fields=fields),
        )

    def check_output_dest(self, predict_id) -> bool:
        """Check if the output destination exists.

        :param predict_id: The identifier of the prediction.
        """
        try:
            self.conn.table(f"{CFG.output_prefix}{predict_id}")
            return True
        except (NoSuchTableError, ibis.IbisError):
            return False

    def create_table_and_schema(self, identifier: str, schema: Schema):
        """Create a schema in the data-backend.

        :param identifier: The identifier of the table.
        :param mapping: The mapping of the schema.
        """
        mapping = convert_schema_to_fields(schema)
        if "id" not in mapping:
            mapping["id"] = "string"
        try:
            mapping = self.db_helper.process_schema_types(mapping)
            t = self.conn.create_table(identifier, schema=ibis.schema(mapping))
        except Exception as e:
            if "exists" in str(e) or "override" in str(e):
                warn("Table already exists, skipping...")
                t = self.conn.table(identifier)
            else:
                raise e

        return t

    def drop(self, force: bool = False):
        """Drop tables or collections in the database.

        :param force: Whether to force the drop.
        """
        if not force and not click.confirm("Are you sure you want to drop all tables?"):
            logging.info("Aborting drop tables")
            return

        for table in self.conn.list_tables():
            logging.info(f"Dropping table: {table}")
            self.drop_table_or_collection(table)

    def get_table_or_collection(self, identifier):
        """Get a table or collection from the database.

        :param identifier: The identifier of the table or collection.
        """
        try:
            return self.conn.table(identifier)
        except ibis.common.exceptions.IbisError:
            raise exceptions.TableNotFoundError(
                f'Table {identifier} not found in database'
            )

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

        return infer_schema(data, identifier=identifier)
