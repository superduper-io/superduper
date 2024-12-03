import os
import typing as t

import click
import mongomock
import pymongo
import pymongo.collection
from superduper import CFG, logging
from superduper.backends.base.data_backend import BaseDataBackend
from superduper.backends.base.metadata import MetaDataStoreProxy
from superduper.base.enums import DBType
from superduper.components.datatype import BaseDataType
from superduper.components.schema import Schema
from superduper.misc.colors import Colors

from superduper_mongodb.artifacts import MongoArtifactStore
from superduper_mongodb.metadata import MongoMetaDataStore
from superduper_mongodb.utils import connection_callback

from .query import MongoQuery


class MongoDBDataBackend(BaseDataBackend):
    """
    Data backend for MongoDB.

    :param uri: URI to the databackend database.
    :param flavour: Flavour of the databackend.
    """

    db_type = DBType.MONGODB

    id_field = "_id"

    def __init__(self, uri: str, flavour: t.Optional[str] = None):
        self.connection_callback = lambda: connection_callback(uri, flavour)
        self.overwrite = True
        super().__init__(uri, flavour=flavour)
        self.conn, self.name = connection_callback(uri, flavour)

        self._db = self.conn[self.name]

        self.datatype_presets = {
            'vector': 'superduper.components.datatype.NativeVector'
        }

    def reconnect(self):
        """Reconnect to mongodb store."""
        # Reconnect to database.
        conn, _ = self.connection_callback()
        self.conn = conn
        self._db = self.conn[self.name]

    def get_query_builder(self, collection_name):
        """Get the query builder for the data backend.

        :param collection_name: Which collection to get the query builder for
        """
        item_gotten = self._db[collection_name]
        if isinstance(
            item_gotten,
            (pymongo.collection.Collection, mongomock.collection.Collection),
        ):
            return MongoQuery(table=collection_name, db=self.datalayer)
        return item_gotten

    def url(self):
        """Return the data backend connection url."""
        return self.conn.HOST + ":" + str(self.conn.PORT) + "/" + self.name

    @property
    def db(self):
        """Return the datalayer instance."""
        return self._db

    def build_metadata(self):
        """Build the metadata store for the data backend."""
        return MetaDataStoreProxy(MongoMetaDataStore(callback=self.connection_callback))

    def build_artifact_store(self):
        """Build the artifact store for the data backend."""
        from mongomock import MongoClient as MockClient

        if isinstance(self.conn, MockClient):
            from superduper.backends.local.artifacts import (
                FileSystemArtifactStore,
            )

            os.makedirs(f"/tmp/{self.name}", exist_ok=True)
            return FileSystemArtifactStore(f"/tmp/{self.name}")
        return MongoArtifactStore(self.conn, f"_filesystem:{self.name}")

    def drop_outputs(self):
        """Drop all outputs."""
        for collection in self.db.list_collection_names():
            if collection.startswith(CFG.output_prefix):
                self.db.drop_collection(collection)

    def drop_table_or_collection(self, name: str):
        """Drop the table or collection.

        Please use with caution as you will lose all data.
        :param name: Collection to drop.
        """
        return self.db.drop_collection(name)

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
        return self.db.client.drop_database(self.db.name)

    def get_table_or_collection(self, identifier):
        """Get a table or collection from the data backend.

        :param identifier: table or collection identifier
        """
        return self._db[identifier]

    def list_tables_or_collections(self):
        """List all tables or collections in the data backend."""
        return self.db.list_collection_names()

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

    def exists(self, table_or_collection, id, key):
        """Check if a document exists in the data backend.

        :param table_or_collection: table or collection identifier
        :param id: document identifier
        :param key: key to check
        """
        return (
            self.db[table_or_collection].find_one(
                {"_id": id, f"{key}._content.bytes": {"$exists": 1}}
            )
            is not None
        )

    def check_output_dest(self, predict_id) -> bool:
        """Check if the output destination exists.

        :param predict_id: identifier of the prediction
        """
        return self.db[f"{CFG.output_prefix}{predict_id}"].find_one() is not None

    def check_ready_ids(
        self,
        query: MongoQuery,
        keys: t.List[str],
        ids: t.Optional[t.List[t.Any]] = None,
    ):
        """Check if all the keys are ready in the ids.

        Use this function to check if all the keys are ready in the ids.
        Because the join operation is not very efficient in MongoDB, we use the
        output keys to filter the ids first and then check the base keys.

        This process only verifies the key and does not involve reading the real data.

        :param query: The query object.
        :param keys: The keys to check.
        :param ids: The ids to check.
        """

        def is_output_key(key):
            return key.startswith(CFG.output_prefix) and key != query.table

        output_keys = [key for key in keys if is_output_key(key)]
        input_ids = ready_ids = ids

        # Filter the ids by the output keys first
        for output_key in output_keys:
            filter: dict[str, t.Any] = {}
            filter[output_key] = {"$exists": 1}
            if ready_ids is not None:
                filter["_source"] = {"$in": ready_ids}
            ready_ids = list(
                self.get_table_or_collection(output_key).find(filter, {"_source": 1})
            )
            ready_ids = [doc["_source"] for doc in ready_ids]
            if not ready_ids:
                return []

        # If we get the ready ids from the output keys, we can continue on these ids
        ids = ready_ids or ids

        base_keys = [key for key in keys if not is_output_key(key)]
        base_filter: dict[str, t.Any] = {}
        base_filter.update({key: {"$exists": 1} for key in base_keys})
        if ready_ids is not None:
            base_filter["_id"] = {"$in": ready_ids}

        ready_ids = list(
            self.get_table_or_collection(query.table).find(base_filter, {"_id": 1})
        )
        ready_ids = [doc["_id"] for doc in ready_ids]

        if ids is not None:
            ready_ids = [id for id in ids if id in ready_ids]

        self._log_check_ready_ids_message(input_ids, ready_ids)
        return ready_ids

    @staticmethod
    def infer_schema(data: t.Mapping[str, t.Any], identifier: t.Optional[str] = None):
        """Infer a schema from a given data object.

        :param data: The data object
        :param identifier: The identifier for the schema, if None, it will be generated
        :return: The inferred schema
        """
        from superduper.misc.auto_schema import infer_schema

        return infer_schema(data, identifier)

    def create_table_and_schema(self, identifier: str, schema: Schema):
        """Create a table and schema in the data backend.

        :param identifier: The identifier for the table
        :param mapping: The mapping for the schema
        """
        # If the data can be converted to JSON,
        # then save it as native data in MongoDB.
        pass
