import os
import typing as t

import click
import mongomock
import pymongo
import pymongo.collection
from superduper import CFG, logging
from superduper.backends.base.data_backend import BaseDataBackend
from superduper.backends.base.metadata import MetaDataStoreProxy
from superduper.components.datatype import BaseDataType
from superduper.components.schema import Schema
from superduper.misc.colors import Colors

from superduper_mongodb.artifacts import MongoDBArtifactStore
from superduper_mongodb.metadata import MongoDBMetaDataStore
from superduper_mongodb.utils import connection_callback


class MongoDBDataBackend(BaseDataBackend):
    """
    Data backend for MongoDB.

    :param uri: URI to the databackend database.
    :param flavour: Flavour of the databackend.
    """

    id_field = "_id"

    def __init__(self, uri: str, plugin: t.Any, flavour: t.Optional[str] = None):
        self.connection_callback = lambda: connection_callback(uri, flavour)

        super().__init__(uri, flavour=flavour, plugin=plugin)

        self.conn, self.name = connection_callback(uri, flavour)
        self._db = self.conn[self.name]

        self.datatype_presets = {
            'vector': 'superduper.components.datatype.NativeVector'
        }

    def reconnect(self):
        """Reconnect to MongoDB databackend."""
        conn, _ = self.connection_callback()
        self.conn = conn
        self._db = self.conn[self.name]

    def build_metadata(self):
        """Build the metadata store for the data backend."""
        return MetaDataStoreProxy(MongoDBMetaDataStore(callback=self.connection_callback))

    def build_artifact_store(self):
        """Build the artifact store for the data backend."""
        from mongomock import MongoClient as MockClient

        if isinstance(self.conn, MockClient):
            from superduper.backends.local.artifacts import (
                FileSystemArtifactStore,
            )

            os.makedirs(f"/tmp/{self.name}", exist_ok=True)
            return FileSystemArtifactStore(f"/tmp/{self.name}")
        return MongoDBArtifactStore(self.conn, f"_filesystem:{self.name}")

    def drop_outputs(self):
        """Drop all outputs."""
        for collection in self._db.list_collection_names():
            if collection.startswith(CFG.output_prefix):
                self._db.drop_collection(collection)

    def drop_table(self, name: str):
        """Drop the table or collection.

        Please use with caution as you will lose all data.
        :param name: Collection to drop.
        """
        return self._db.drop_collection(name)

    def drop(self, force: bool = False):
        """Drop the data backend.

        Please use with caution as you will lose all data.
        :param force: Force the drop, default is False.
                      If False, a confirmation prompt will be displayed.
        """
        if not force:
            if not click.confirm(
                f"{Colors.RED}[!!!WARNING USE WITH CAUTION AS YOU "
                f"WILL LOSE ALL DATA!!!]{Colors.RESET} "
                "Are you sure you want to drop the data-backend? ",
                default=False,
            ):
                logging.warn("Aborting...")
        return self._db.client.drop_database(self._db.name)

    def get_table(self, identifier):
        """Get a table or collection from the data backend.

        :param identifier: table or collection identifier
        """
        return self._db[identifier]

    def list_tables(self):
        """List all tables or collections in the data backend."""
        return self._db.list_collection_names()

    def disconnect(self):
        """Disconnect the client."""
        # TODO: implement me

    def create_output_dest(
        self,
        predict_id: str,
        datatype: t.Union[str, BaseDataType],
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
        return self._db[f"{CFG.output_prefix}{predict_id}"].find_one() is not None

    def create_table_and_schema(self, identifier: str, schema: Schema):
        """Create a table and schema in the data backend.

        :param identifier: The identifier for the table
        :param mapping: The mapping for the schema
        """
        # If the data can be converted to JSON,
        # then save it as native data in MongoDB.
        pass
