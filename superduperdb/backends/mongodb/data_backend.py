import os
import typing as t

import click
import pymongo

from superduperdb import logging
from superduperdb.backends.base.data_backend import BaseDataBackend
from superduperdb.backends.ibis.field_types import FieldType
from superduperdb.backends.mongodb.artifacts import MongoArtifactStore
from superduperdb.backends.mongodb.metadata import MongoMetaDataStore
from superduperdb.base.enums import DBType
from superduperdb.base.serializable import Serializable
from superduperdb.components.datatype import DataType
from superduperdb.misc.colors import Colors
from superduperdb.misc.special_dicts import MongoStyleDict


class MongoDataBackend(BaseDataBackend):
    """
    Data backend for MongoDB.

    :param conn: MongoDB client connection
    :param name: Name of database to host filesystem
    """

    db_type = DBType.MONGODB

    id_field = '_id'

    def __init__(self, conn: pymongo.MongoClient, name: str):
        super().__init__(conn=conn, name=name)
        self._db = self.conn[self.name]

    def url(self):
        """Return the data backend connection url."""
        return self.conn.HOST + ':' + str(self.conn.PORT) + '/' + self.name

    @property
    def db(self):
        """Return the datalayer instance."""
        return self._db

    def build_metadata(self):
        """Build the metadata store for the data backend."""
        return MongoMetaDataStore(self.conn, self.name)

    def build_artifact_store(self):
        """Build the artifact store for the data backend."""
        from mongomock import MongoClient as MockClient

        if isinstance(self.conn, MockClient):
            from superduperdb.backends.local.artifacts import (
                FileSystemArtifactStore,
            )

            os.makedirs(f'/tmp/{self.name}', exist_ok=True)
            return FileSystemArtifactStore(f'/tmp/{self.name}')
        return MongoArtifactStore(self.conn, f'_filesystem:{self.name}')

    def drop(self, force: bool = False):
        """Drop the data backend.

        Please use with caution as you will lose all data.
        :param force: Force the drop, default is False.
                      If False, a confirmation prompt will be displayed.
        """
        if not force:
            if not click.confirm(
                f'{Colors.RED}[!!!WARNING USE WITH CAUTION AS YOU '
                f'WILL LOSE ALL DATA!!!]{Colors.RESET} '
                'Are you sure you want to drop the data-backend? ',
                default=False,
            ):
                logging.warn('Aborting...')
        return self.db.client.drop_database(self.db.name)

    def get_table_or_collection(self, identifier):
        """Get a table or collection from the data backend.

        :param identifier: table or collection identifier
        """
        return self._db[identifier]

    def set_content_bytes(self, r, key, bytes_):
        """Set the content bytes in the data backend.

        :param r: dictionary containing information about the content
        :param key: key to set
        :param bytes_: content bytes
        """
        if not isinstance(r, MongoStyleDict):
            r = MongoStyleDict(r)
        r[f'{key}._content.bytes'] = bytes_
        return r

    def exists(self, table_or_collection, id, key):
        """Check if a document exists in the data backend.

        :param table_or_collection: table or collection identifier
        :param id: document identifier
        :param key: key to check
        """
        return (
            self.db[table_or_collection].find_one(
                {'_id': id, f'{key}._content.bytes': {'$exists': 1}}
            )
            is not None
        )

    # TODO: Remove the unused function
    def unset_outputs(self, info: t.Dict):
        """Unset the output field in the data backend.

        :param info: dictionary containing information about the output field
        """
        select = Serializable.from_dict(info['select'])
        logging.info(f'unsetting output field _outputs.{info["key"]}.{info["model"]}')
        doc = {'$unset': {f'_outputs.{info["key"]}.{info["model"]}': 1}}
        update = select.update(doc)
        return self.db[select.collection].update_many(update.filter, update.update)

    def list_vector_indexes(self):
        """List all vector indexes in the data backend."""
        indexes = []
        for coll in self.db.list_collection_names():
            i = self.db.command({'listSearchIndexes': coll})
            try:
                batch = i['cursor']['firstBatch'][0]
            except IndexError:
                continue
            if '_outputs' in batch['latestDefinition']['mappings']['fields']:
                indexes.append(batch['name'])
        return indexes

    def list_tables_or_collections(self):
        """List all tables or collections in the data backend."""
        return self.db.list_collection_names()

    def delete_vector_index(self, vector_index):
        """
        Delete a vector index in the data backend if an Atlas deployment.

        :param vector_index: vector index to delete
        """
        # see `VectorIndex` class for details
        # indexing_listener contains a `Select` object
        assert not isinstance(vector_index.indexing_listener, str)
        select = vector_index.indexing_listener.select

        # TODO: probably MongoDB queries should all have a common base class
        self.db.command(
            {
                "dropSearchIndex": select.table_or_collection.identifier,
                "name": vector_index.identifier,
            }
        )

    def disconnect(self):
        """Disconnect the client."""

        # TODO: implement me

    def create_output_dest(
        self,
        predict_id: str,
        datatype: t.Union[None, DataType, FieldType],
        flatten: bool = False,
    ):
        """Create an output collection for a component.

        That will do nothing for MongoDB.

        :param predict_id: The predict id of the output destination
        :param datatype: datatype of component
        :param flatten: flatten the output
        """
        pass

    def check_output_dest(self, predict_id) -> bool:
        """Check if the output destination exists.

        :param predict_id: identifier of the prediction
        """
        return True

    @staticmethod
    def infer_schema(data: t.Mapping[str, t.Any], identifier: t.Optional[str] = None):
        """Infer a schema from a given data object.

        :param data: The data object
        :param identifier: The identifier for the schema, if None, it will be generated
        :return: The inferred schema
        """
        from superduperdb.misc.auto_schema import infer_schema

        return infer_schema(data, identifier)
