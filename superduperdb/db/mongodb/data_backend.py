import os
import re
import typing as t

import click
import pymongo

from superduperdb import logging
from superduperdb.container.serializable import Serializable
from superduperdb.container.vector_index import VectorIndex
from superduperdb.db.base.data_backend import BaseDataBackend
from superduperdb.db.mongodb.artifacts import MongoArtifactStore
from superduperdb.db.mongodb.metadata import MongoMetaDataStore
from superduperdb.misc.colors import Colors
from superduperdb.misc.special_dicts import MongoStyleDict


class MongoDataBackend(BaseDataBackend):
    """
    Data backend for MongoDB.

    :param conn: MongoDB client connection
    :param name: Name of database to host filesystem
    """

    id_field = '_id'

    def __init__(self, conn: pymongo.MongoClient, name: str):
        super().__init__(conn=conn, name=name)
        self._db = self.conn[self.name]

    @property
    def db(self):
        return self._db

    def build_metadata(self):
        return MongoMetaDataStore(self.conn, self.name)

    def build_artifact_store(self):
        from mongomock import MongoClient as MockClient

        if isinstance(self.conn, MockClient):
            from superduperdb.db.filesystem.artifacts import FileSystemArtifactStore

            os.makedirs(f'/tmp/{self.name}', exist_ok=True)
            return FileSystemArtifactStore(f'/tmp/{self.name}')
        return MongoArtifactStore(self.conn, f'_filesystem:{self.name}')

    def drop(self, force: bool = False):
        if not force:
            if not click.confirm(
                f'{Colors.RED}[!!!WARNING USE WITH CAUTION AS YOU '
                f'WILL LOSE ALL DATA!!!]{Colors.RESET} '
                'Are you sure you want to drop the data-backend? ',
                default=False,
            ):
                print('Aborting...')
        return self.db.client.drop_database(self.db.name)

    def set_content_bytes(self, r, key, bytes_):
        if not isinstance(r, MongoStyleDict):
            r = MongoStyleDict(r)
        r[f'{key}._content.bytes'] = bytes_
        return r

    def unset_outputs(self, info: t.Dict):
        select = Serializable.deserialize(info['select'])
        logging.info(f'unsetting output field _outputs.{info["key"]}.{info["model"]}')
        doc = {'$unset': {f'_outputs.{info["key"]}.{info["model"]}': 1}}
        update = select.update(doc)
        return self.db[select.collection].update_many(update.filter, update.update)

    def list_vector_indexes(self):
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

    def delete_vector_index(self, vector_index: VectorIndex):
        """
        Delete a vector index in the data backend if an Atlas deployment.

        :param vector_index: vector index to delete
        """
        # see `VectorIndex` class for details
        # indexing_listener contains a `Select` object
        assert not isinstance(vector_index.indexing_listener, str)
        select = vector_index.indexing_listener.select

        # TODO: probably MongoDB queries should all have a common base class
        collection = select.collection  # type: ignore[attr-defined]
        self.db.command(
            {
                "dropSearchIndex": collection.name,
                "name": vector_index.identifier,
            }
        )

    def create_vector_index(self, vector_index):
        """
        Create a vector index in the data backend if an Atlas deployment.

        :param vector_index: vector index to create
        """
        collection = vector_index.indexing_listener.select.collection.name
        key = vector_index.indexing_listener.key
        if re.match('^_outputs\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+$', key):
            key = key.split('.')[1]
        model = vector_index.indexing_listener.model.identifier
        fields = {
            model: [
                {
                    "dimensions": vector_index.dimensions,
                    "similarity": vector_index.measure,
                    "type": "knnVector",
                }
            ]
        }
        self.db.command(
            {
                "createSearchIndexes": collection,
                "indexes": [
                    {
                        "name": vector_index.identifier,
                        "definition": {
                            "mappings": {
                                "dynamic": True,
                                "fields": {
                                    "_outputs": {
                                        "fields": {
                                            key: {
                                                "fields": fields,
                                                "type": "document",
                                            }
                                        },
                                        "type": "document",
                                    }
                                },
                            }
                        },
                    }
                ],
            }
        )
