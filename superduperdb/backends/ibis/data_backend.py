import base64
import typing as t
from warnings import warn

import ibis
import pandas
from ibis.backends.base import BaseBackend

from superduperdb.backends.base.data_backend import BaseDataBackend
from superduperdb.backends.ibis.db_helper import get_insert_processor
from superduperdb.backends.ibis.field_types import FieldType, dtype
from superduperdb.backends.ibis.query import Table
from superduperdb.backends.ibis.utils import get_output_table_name
from superduperdb.backends.local.artifacts import FileSystemArtifactStore
from superduperdb.backends.sqlalchemy.metadata import SQLAlchemyMetadata
from superduperdb.components.model import APIModel, Model
from superduperdb.components.schema import Schema

BASE64_PREFIX = 'base64:'


class IbisDataBackend(BaseDataBackend):
    def __init__(self, conn: BaseBackend, name: str, in_memory: bool = False):
        super().__init__(conn=conn, name=name)
        self.in_memory = in_memory

    def url(self):
        return self.conn.con.url + self.name

    def build_artifact_store(self):
        return FileSystemArtifactStore(conn='.superduperdb/artifacts/', name='ibis')

    def build_metadata(self):
        return SQLAlchemyMetadata(conn=self.conn.con, name='ibis')

    def create_ibis_table(self, identifier: str, schema: Schema):
        self.conn.create_table(identifier, schema=schema)

    def insert(self, table_name, raw_documents):
        for doc in raw_documents:
            for k, v in doc.items():
                doc[k] = self.convert_data_format(v)
        table_name, raw_documents = get_insert_processor(self.conn.name)(
            table_name, raw_documents
        )
        if not self.in_memory:
            self.conn.insert(table_name, raw_documents)
        else:
            self.conn.create_table(table_name, pandas.DataFrame(raw_documents))

    @staticmethod
    def convert_data_format(data):
        """Convert byte data to base64 format for storage in the database."""

        if isinstance(data, bytes):
            return BASE64_PREFIX + base64.b64encode(data).decode('utf-8')
        else:
            return data

    @staticmethod
    def recover_data_format(data):
        """Recover byte data from base64 format stored in the database."""
        if isinstance(data, str) and data.startswith(BASE64_PREFIX):
            return base64.b64decode(data[len(BASE64_PREFIX) :])
        else:
            return data

    def create_model_table_or_collection(self, model: t.Union[Model, APIModel]):
        msg = (
            "Model must have an encoder to create with the"
            f" {type(self).__name__} backend."
        )
        assert model.encoder is not None, msg
        if isinstance(model.encoder, FieldType):
            output_type = dtype(model.encoder.identifier)
        else:
            output_type = model.encoder
        fields = {
            'output_id': dtype('int32'),
            'input_id': dtype('str'),
            'query_id': dtype('string'),
            'output': output_type,
            'key': dtype('string'),
        }
        return Table(
            identifier=get_output_table_name(model.identifier, model.version),
            schema=Schema(
                identifier=f'_schema/{model.identifier}/{model.version}', fields=fields
            ),
        )

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
        raise NotImplementedError(
            "Dropping tables needs to be done in each DB natively"
        )

    def get_table_or_collection(self, identifier):
        return self.conn.table(identifier)

    def disconnect(self):
        """
        Disconnect the client
        """

        # TODO: implement me
