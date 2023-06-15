import typing as t

from bson import ObjectId
from pymongo import UpdateOne
from pymongo.cursor import Cursor

from superduperdb.core.documents import Document
from superduperdb.datalayer.base.data_backend import BaseDataBackend
from superduperdb.datalayer.mongodb.query import Delete, Update, Select, Insert
from superduperdb.misc.special_dicts import MongoStyleDict
from superduperdb.misc.logger import logging


class MongoDataBackend(BaseDataBackend):
    select_cls = Select
    id_field = '_id'

    def __init__(
        self,
        conn: t.Any,
        name: t.Optional[str] = None,
    ):
        super().__init__(conn, name)
        self.db = conn[name]

    def insert(self, insert: Insert):
        encoded = [r.encode() for r in insert.documents]
        return self.db[insert.collection].insert_many(
            encoded,
            ordered=insert.ordered,
            bypass_document_validation=insert.bypass_document_validation,
        )

    def download_update(self, table, id, key, bytes):
        return Update(
            collection=table,
            one=True,
            filter={'_id': id},
            update=Document({'$set': {f'{key}._content.bytes': bytes}}),
        )

    def get_ids_from_select(self, select: Select):
        return [
            r['_id'] for r in self.db[select.collection].find(select.filter, {'_id': 1})
        ]

    def get_output_from_document(self, r: Document, key: str, model: str):
        return MongoStyleDict(r.content)[f'_outputs.{key}.{model}'], r.content['_id']

    def get_raw_cursor(self, select: Select):
        return Cursor(
            self.db[select.collection],
            select.filter,
            select.projection,
            **select.kwargs,
        )

    def get_query_for_validation_set(self, validation_set):
        return Select(
            collection='_validation_sets', filter={'identifier': validation_set}
        )

    def insert_validation_data(self, tmp, identifier):
        for i, r in enumerate(tmp):
            tmp[i]['identifier'] = identifier
        self.insert(Insert(collection='_validation_sets', documents=tmp))

    def set_content_bytes(self, r, key, bytes_):
        if not isinstance(r, MongoStyleDict):
            r = MongoStyleDict(r)
        r[f'{key}._content.bytes'] = bytes_
        return r

    def _add_features(self, r):
        r = MongoStyleDict(r)
        for k in self.features:
            r[k] = r['_outputs'][k][self.features[k]]
        if '_other' in r:
            for k in self.features:
                if k in r['_other']:
                    r['_other'][k] = r['_outputs'][k][self.features[k]]
        return r

    def write_outputs(self, watcher_info, outputs, _ids):
        logging.info('bulk writing...')
        select = Select(**watcher_info['select'])
        if watcher_info.get('target') is None:
            out_key = f'_outputs.{watcher_info["key"]}.{watcher_info["model"]}'
        else:
            out_key = watcher_info['target']

        self.db[select.collection].bulk_write(
            [
                UpdateOne(
                    {'_id': ObjectId(id)},
                    {'$set': {out_key: outputs[i]}},
                )
                for i, id in enumerate(_ids)
            ]
        )
        logging.info('done.')

    def update(self, update: Update):
        if update.replacement is not None:
            return self.db[update.collection].replace_one(
                update.filter, update.replacement.encode()
            )
        if update.one:
            return self.db[update.collection].update_one(
                update.filter, update.update.encode()
            )
        return self.db[update.collection].update_many(
            update.filter, update.update.encode()
        )

        pass

    def delete(self, delete: Delete):
        if delete.one:
            return self.db[delete.collection].delete_one(delete.filter)
        else:
            return self.db[delete.collection].delete_many(delete.filter)

    def unset_outputs(self, info: t.Dict):
        select = Select(**info['select'])
        logging.info(f'unsetting output field _outputs.{info["key"]}.{info["model"]}')
        update = select.update(
            {'$unset': {f'_outputs.{info["key"]}.{info["model"]}': 1}}
        )
        return self.db[select.collection].update_many(update.filter, update.update)

    def show_validation_sets(self):
        return self.db['_validation_sets'].distinct('identifier')
