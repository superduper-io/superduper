import typing as t

import click
from pymongo import MongoClient

from superduperdb import logging
from superduperdb.container.document import Document
from superduperdb.container.serializable import Serializable
from superduperdb.container.vector_index import VectorIndex
from superduperdb.db.base.data_backend import BaseDataBackend
from superduperdb.misc.colors import Colors
from superduperdb.misc.special_dicts import MongoStyleDict


class MongoDataBackend(BaseDataBackend):
    id_field = '_id'

    def __init__(self, conn: MongoClient, name: str):
        super().__init__(conn=conn, name=name)
        self._db = self.conn[self.name]

    @property
    def db(self):
        return self._db

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

    def get_output_from_document(
        self, r: Document, key: str, model: str
    ) -> t.Tuple[MongoStyleDict, t.Any]:
        assert isinstance(r.content, dict)
        return (
            MongoStyleDict(r.content)[f'_outputs.{key}.{model}'],
            r.content['_id'],
        )

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
        collection = vector_index.indexing_listener.select.collection.name
        key = vector_index.indexing_listener.key
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
