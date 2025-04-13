import glob
import os
import time
import typing as t

import click
import ibis

import ibis.common
import ibis.common.exceptions
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


class IbisDataBackend(BaseDataBackend):
    """Ibis data backend for the database.

    :param uri: URI to the databackend database.
    :param flavour: Flavour of the databackend.
    """

    db_type = DBType.SQL

    def __init__(self, uri: str, flavour: t.Optional[str] = None):
        self.connection_callback = lambda: self._connection_callback(uri)
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

    @staticmethod
    def _connection_callback(uri):
        name = uri.split("//")[0]
        in_memory = False
        ibis_conn = ibis.connect(uri)
        return ibis_conn, name, in_memory

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

    def _check_token(self):
        import datetime

        auth_token = os.environ['SUPERDUPER_AUTH_TOKEN']
        with open(auth_token) as f:
            expiration_date = datetime.datetime.strptime(
                f.read().split('\n')[0].strip(), "%Y-%m-%d %H:%M:%S.%f"
            )
        if expiration_date < datetime.datetime.now():
            raise Exception("auth token expired")

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
        self.conn.insert(table_name, raw_documents)

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
        except (NoSuchTableError, ibis.IbisError, ibis.common.exceptions.TableNotFound):
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
            t = self._create_table_with_retry(identifier, schema=ibis.schema(mapping))
        except Exception as e:
            if "exists" in str(e) or "override" in str(e):
                logging.warn("Table already exists, skipping...")
                t = self.conn.table(identifier)
            else:
                raise e

        return t

    def _create_table_with_retry(self, table_name, schema, retry=3):
        for attempt in range(retry):
            t = self.conn.create_table(table_name, schema=schema)

            if table_name in self.conn.list_tables():
                return t
            else:
                logging.warn(
                    f"Failed to create table {table_name}. "
                    f"Attempt {attempt + 1}/{retry}"
                )
                time.sleep(1)

        raise exceptions.TableNotFoundError(f"Failed to create table {table_name}")

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
