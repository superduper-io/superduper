from typing import List

from bson import ObjectId
from pymongo import UpdateOne
from pymongo.cursor import Cursor
from pymongo.database import Database as MongoDatabase

import superduperdb.datalayer.mongodb.artifacts
import superduperdb.datalayer.mongodb.collection
from superduperdb.core.documents import Document
from superduperdb.datalayer.base.database import BaseDatabase
from superduperdb.datalayer.mongodb.cursor import SuperDuperCursor
from superduperdb.misc.special_dicts import MongoStyleDict
from superduperdb.misc.logger import logging
from superduperdb.datalayer.mongodb.query import (
    Select,
    Delete,
    Insert,
    Update,
)


class Database(MongoDatabase, BaseDatabase):
    """
    Database building on top of :code:`pymongo.database.Database`. Collections in the
    database are SuperDuperDB objects :code:`superduperdb.collection.Collection`.
    """

    _database_type = 'mongodb'
    select_cls = Select
    id_field = '_id'

    def __init__(self, artifact_store, metadata, *args, **kwargs):
        MongoDatabase.__init__(self, *args, **kwargs)
        BaseDatabase.__init__(
            self,
            artifact_store=artifact_store,
            metadata=metadata,
        )

    def __getitem__(self, name: str):
        if name != '_validation_sets' and name.startswith('_'):
            return super().__getitem__(name)
        return superduperdb.datalayer.mongodb.collection.Collection(self, name)

    def _add_split_to_row(self, r, other):
        r['_other'] = other
        return r

    def _base_delete(self, delete: Delete, **kwargs):
        return self[delete.collection]._base_delete(delete)

    def _base_insert(self, insert: Insert):
        return self[insert.collection]._base_insert_many(insert)

    def _base_update(self, update: Update):
        return self[update.collection]._base_update(update)

    def _convert_id_to_str(self, id_):
        return str(id_)

    def _convert_str_to_id(self, id_):
        return ObjectId(id_)

    def _download_update(self, table, id, key, bytes_):
        return Update(
            collection=table,
            one=True,
            filter={'_id': id},
            update=Document({'$set': {f'{key}._content.bytes': bytes_}}),
        )

    def _get_cursor(self, select: Select, features=None, scores=None):
        return SuperDuperCursor(
            self[select.collection],
            select.filter,
            select.projection,
            features=features,
            **select.kwargs,
        )

    def _get_output_from_document(self, r: Document, key: str, model: str):
        return MongoStyleDict(r.content)[f'_outputs.{key}.{model}'], r.content['_id']

    def _get_ids_from_select(self, select: Select):
        return [
            r['_id']
            for r in self[select.collection]._base_find(select.filter, {'_id': 1})
        ]

    def _get_raw_cursor(self, select: Select):
        return Cursor(
            self[select.collection], select.filter, select.projection, **select.kwargs
        )

    def get_query_for_validation_set(self, validation_set):
        return Select(
            collection='_validation_sets', filter={'identifier': validation_set}
        )

    def _insert_validation_data(self, tmp: List[Document], identifier: str):
        for i, r in enumerate(tmp):
            tmp[i]['identifier'] = identifier
        self._insert(Insert(collection='_validation_sets', documents=tmp))

    def _show_validation_sets(self):
        return self['_validation_sets'].distinct('identifier')

    def _separate_query_part_from_validation_record(self, r):
        return r['_other'], {k: v for k, v in r.items() if k != '_other'}

    def _set_content_bytes(self, r, key, bytes_):
        if not isinstance(r, MongoStyleDict):
            r = MongoStyleDict(r)
        r[f'{key}._content.bytes'] = bytes_
        return r

    def _unset_watcher_outputs(self, info):
        select = self.select_cls(**info['_select'])
        logging.info(f'unsetting output field _outputs.{info["key"]}.{info["model"]}')
        return self._base_update(
            select.update({'$unset': {f'_outputs.{info["key"]}.{info["model"]}': 1}})
        )

    def _write_watcher_outputs(self, watcher_info, outputs, _ids):
        logging.info('bulk writing...')
        select = self.select_cls(**watcher_info['_select'])
        if watcher_info.get('target') is None:
            out_key = f'_outputs.{watcher_info["key"]}.{watcher_info["model"]}'
        else:
            out_key = watcher_info['target']

        self[select.collection].bulk_write(
            [
                UpdateOne(
                    {'_id': ObjectId(id)},
                    {'$set': {out_key: outputs[i]}},
                )
                for i, id in enumerate(_ids)
            ]
        )
        logging.info('done.')
