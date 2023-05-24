from typing import Union

import gridfs
from bson import ObjectId
from pymongo import UpdateOne
from pymongo.cursor import Cursor
from pymongo.database import Database as MongoDatabase

import superduperdb.datalayer.mongodb.collection
from superduperdb.datalayer.base.database import BaseDatabase
from superduperdb.cluster.annotations import ObjectIdConvertible, List
from superduperdb.cluster.job_submission import work
from superduperdb.datalayer.mongodb import loading
from superduperdb.datalayer.mongodb.cursor import SuperDuperCursor
from superduperdb.misc.special_dicts import MongoStyleDict
from superduperdb.misc.logger import logging
from superduperdb.datalayer.mongodb.query import Select, Delete, Insert, set_one_key_in_document, \
    Update


class Database(MongoDatabase, BaseDatabase):
    """
    Database building on top of :code:`pymongo.database.Database`. Collections in the
    database are SuperDuperDB objects :code:`superduperdb.collection.Collection`.
    """

    _id = '_id'
    _database_type = 'mongodb'
    select_cls = Select

    def __init__(self, *args, **kwargs):
        MongoDatabase.__init__(self, *args, **kwargs)
        BaseDatabase.__init__(self)
        self._filesystem = None
        self._filesystem_name = f'_{self.name}:files'

    def __getitem__(self, name: str):
        if name != '_validation_sets' and name.startswith('_'):
            return super().__getitem__(name)
        return superduperdb.datalayer.mongodb.collection.Collection(self, name)

    @property
    def filesystem(self):
        if self._filesystem is None:
            self._filesystem = gridfs.GridFS(self.client[self._filesystem_name])
        return self._filesystem

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

    def _create_job_record(self, r):
        self['_jobs'].insert_one(r)

    def _create_object_entry(self, info):
        return self['_objects'].insert_one(info)

    def _delete_object_info(self, identifier, variety):
        return self['_objects'].delete_one(
            {'identifier': identifier, 'variety': variety}
        )

    def _download_update(self, table, id, key, bytes_):
        return Update(
            collection=table,
            one=True,
            filter={'_id': id},
            update={'$set': {f'{key}._content.bytes': bytes_}},
        )

    def _get_cursor(self, select: Select, features=None, scores=None):
        return SuperDuperCursor(
            self[select.collection],
            select.filter,
            select.projection,
            features=features,
            **select.kwargs,
        )

    def _get_hash_from_document(self, r, watcher_info):
        return MongoStyleDict(r)[f'_outputs.{watcher_info["key"]}.{watcher_info["model"]}']

    def _get_ids_from_select(self, select: Select):
        return [r['_id'] for r in self[select.collection]._base_find(select.filter, {'_id': 1})]

    def _get_job_info(self, identifier):
        return self['_jobs'].find_one({'identifier': identifier})

    def get_meta_data(self, **kwargs):
        return self['_meta'].find_one(kwargs)['value']

    def _get_object_info(self, identifier, variety, **kwargs):
        return self['_objects'].find_one({'identifier': identifier, 'variety': variety, **kwargs})

    def _get_object_info_where(self, variety, **kwargs):
        return self['_objects'].find_one({'variety': variety, **kwargs})

    def _get_raw_cursor(self, select: Select):
        return Cursor(
            self[select.collection],
            select.filter,
            select.projection,
            **select.kwargs
        )

    def get_query_for_validation_set(self, validation_set):
        return Select(collection='_validation_sets', filter={'identifier': validation_set})

    def _insert_validation_data(self, tmp, identifier):
        tmp = [{**r, 'identifier': identifier} for r in tmp]
        self['_validation_sets'].insert_many(tmp)

    def list_jobs(self, status=None):
        status = {} if status is None else {'status': status}
        return list(
            self['_jobs'].find(
                status, {'identifier': 1, '_id': 0, 'method': 1, 'status': 1, 'time': 1}
            )
        )

    def _list_objects(self, variety, **kwargs):
        return self['_objects'].distinct('identifier', {'variety': variety, **kwargs})

    def list_validation_sets(self):
        return self['_validation_sets'].distinct('identifier')

    def _load_blob_of_bytes(self, file_id):
        return loading.load(file_id, filesystem=self.filesystem)

    def _replace_object(self, file_id, new_file_id, variety, identifier):
        self.filesystem.delete(file_id)
        self['_objects'].update_one(
            {
                'identifier': identifier,
                'variety': variety,
            },
            {'$set': {'object': new_file_id}},
        )

    def _save_blob_of_bytes(self, bytes_):
        return loading.save(bytes_, self.filesystem)

    def save_metrics(self, identifier, variety, metrics):
        self['_objects'].update_one(
            {'identifier': identifier, 'variety': variety},
            {'$set': {'metric_values': metrics}},
        )

    def separate_query_part_from_validation_record(self, r):
        return r['_other'], {k: v for k, v in r.items() if k != '_other'}

    def _set_content_bytes(self, r, key, bytes_):
        if not isinstance(r, MongoStyleDict):
            r = MongoStyleDict(r)
        r[f'{key}._content.bytes'] = bytes_
        return r

    def set_job_flag(self, identifier, kw):
        self['_jobs'].update_one({'identifier': identifier}, {'$set': {kw[0]: kw[1]}})

    def _unset_neighbourhood_data(self, info, watcher_info):
        select = self.select_cls(**watcher_info['select'])
        logging.info(f'unsetting neighbourhood {info["info"]}')
        update = select.update({'$unset': {f'_like.{info["identifier"]}': 1}})
        return self._base_update(update)

    def _unset_watcher_outputs(self, info):
        select = self.select_cls(**info['select'])
        logging.info(f'unsetting output field _outputs.{info["key"]}.{info["model"]}')
        return self._base_update(
            select.update({'$unset': {f'_outputs.{info["key"]}.{info["model"]}': 1}})
        )

    def _update_neighbourhood(self, ids, similar_ids, identifier, select: Select):
        self[select.collection].bulk_write([
            UpdateOne({'_id': id_}, {'$set': {f'_like.{identifier}': sids}})
            for id_, sids in zip(ids, similar_ids)
        ])

    def _update_job_info(self, identifier, key, value):
        self['_jobs'].update_one({'identifier': identifier}, {'$set': {key: value}})

    def _update_object_info(self, identifier, variety, key, value):
        self['_objects'].update_one(
            {'identifier': identifier, 'variety': variety}, {'$set': {key: value}}
        )

    def write_output_to_job(self, identifier, msg, stream):
        assert stream in {'stdout', 'stderr'}
        self['_jobs'].update_one({'identifier': identifier}, {'$push': {stream: msg}})

    def _write_watcher_outputs(self, watcher_info, outputs, _ids):
        logging.info('bulk writing...')
        select = self.select_cls(**watcher_info['select'])
        if watcher_info.get('target') is None:
            self[select.collection].bulk_write([
                UpdateOne(
                    {'_id': id},
                    {'$set': {
                        f'_outputs.{watcher_info["key"]}.{watcher_info["model"]}': outputs[i]}
                    },
                )
                for i, id in enumerate(_ids)
            ])
        else:  # pragma: no cover
            self[select.collection].bulk_write([
                UpdateOne(
                    {'_id': id}, {'$set': {watcher_info['target']: outputs[i]}}
                )
                for i, id in enumerate(_ids)
            ])
        logging.info('done.')

