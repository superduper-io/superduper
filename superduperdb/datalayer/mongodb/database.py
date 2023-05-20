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


class Database(MongoDatabase, BaseDatabase):
    """
    Database building on top of :code:`pymongo.database.Database`. Collections in the
    database are SuperDuperDB objects :code:`superduperdb.collection.Collection`.
    """

    _id = '_id'
    _database_type = 'mongodb'

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

    def apply_agent(self, agent, filter_=None, projection=None, like=None):
        query_params = (filter_ or {}, projection or {})
        return self.apply_agent(agent, query_params, like=like)

    def _base_delete(self, method, collection, filter_=None, *args, **kwargs):
        assert method == 'delete_many'
        filter_ = filter_ or {}
        self[collection]._base_delete(filter_, *args, **kwargs)

    def _base_insert(self, items, *args, **kwargs):
        return self[args[1]]._base_insert_many(items, *args[2:], **kwargs)

    def _base_update(self, *args, **kwargs):
        op, collection, filter_, update = args[:4]
        args = args[4:]
        return self[collection]._base_update(op, filter_, update, *args, **kwargs)

    def _classify_query(self, method, *args, **kwargs):
        if method == 'find':
            return 'SELECT'
        elif method == 'insert_many':
            return 'INSERT'
        elif method in {'replace_one', 'update_many'}:
            return 'UPDATE'
        elif method == 'delete_many':
            return 'DELETE'
        else:
            raise NotImplementedError(f'{method} not recognized.')

    def _convert_query_params_to_bytes(self, method, collection, filter_, update, *args, **kwargs):
        update = self.convert_from_types_to_bytes(update)
        return (method, collection, filter_, update, *args), kwargs

    def _convert_id_to_str(self, id_):
        return str(id_)

    def _convert_str_to_id(self, id_):
        return ObjectId(id_)

    def _create_job_record(self, r):
        self['_jobs'].insert_one(r)

    def _create_object_entry(self, info):
        return self['_objects'].insert_one(info)

    def create_watcher(
        self,
        collection,
        model,
        filter_=None,
        projection=None,
        key='_base',
        verbose=False,
        target=None,
        process_docs=True,
        features=None,
        loader_kwargs=None,
        dependencies=(),
    ):
        query_params = ['find', collection]
        if filter_ is None:
            query_params.append({})
        else:
            query_params.append(filter_)
        if projection is not None:
            query_params.append(projection)

        return self._create_watcher(
            f'{model}/{key}',
            model,
            query_params=query_params,
            key=key,
            verbose=verbose,
            target=target,
            process_docs=process_docs,
            features=features,
            predict_kwargs=loader_kwargs,
            dependencies=dependencies,
        )

    def create_validation_set(
        self, identifier, collection, filter_, chunk_size=1000, sample_size=None
    ):
        if filter_ is None:
            filter_ = {}
        if sample_size is not None:
            sample = self[collection].aggregate(
                [
                    {'$match': filter_},
                    {'$sample': {'size': sample_size}},
                    {'$project': {'_id': 1}},
                ]
            )
            _ids = [r['_id'] for r in sample]
            filter_['_id'] = {'$in': _ids}
        return super()._create_validation_set(
            identifier, 'find', collection, filter_, chunk_size=chunk_size
        )

    def _delete_object_info(self, identifier, variety):
        return self['_objects'].delete_one(
            {'identifier': identifier, 'variety': variety}
        )

    @work
    def download_content(
        self,
        query_params,
        ids: List(ObjectIdConvertible()) = None,
        documents=None,
        timeout=None,
        raises=True,
        n_download_workers=None,
        headers=None,
        **kwargs,
    ):
        return super().download_content(
            query_params,
            ids=ids,
            documents=documents,
            timeout=None,
            raises=raises,
            n_download_workers=n_download_workers,
            headers=headers,
            remote=False,
            **kwargs,
        )

    def _download_update(self, collection, _id, key, bytes_):
        self[collection].update_one(
            {'_id': _id}, {'$set': {f'{key}._content.bytes': bytes_}}, refresh=False
        )

    def _execute_query_to_get_hashes(self, *query_params):
        query_params = list(query_params)
        if len(query_params) >= 3:
            query_params[2]['_id'] = 1
        return self.execute_query(*query_params)

    def _format_fold_to_query(self, query_params, fold):
        if not query_params[2:]:
            query_params = (*query_params, {})
        query_params = list(query_params)
        query_params[2]['_fold'] = fold
        return tuple(query_params)

    def _get_cursor(self, *args, features=None, **kwargs):
        collection = self[args[1]]
        return SuperDuperCursor(collection, *args[2:], features=features, **kwargs)

    def _get_docs_from_ids(self, _ids, *query_params, features=None, raw=False):
        query_params = list(query_params)
        if query_params[2:]:
            query_params[2] = {'_id': {'$in': _ids}, **query_params[2]}
        else:
            query_params.append({})
        return list(
            self[query_params[1]].find(*query_params[2:], features=features, raw=raw)
        )

    def _get_hash_from_record(self, r, watcher_info):
        return MongoStyleDict(r)[
            f'_outputs.{watcher_info["key"]}.{watcher_info["model"]}'
        ]

    def get_ids_from_result(self, query_params, result):
        _ids = []
        for r in result:
            _ids.append(r['_id'])
        return _ids

    def _get_ids_from_query(self, op, collection, filter_=None, *args, **kwargs):
        filter_ = filter_ or {}
        if op.endswith('_one'):
            _id = self[collection].find_one(filter_, {'_id': 1})['_id']
            return [_id]
        else:
            _ids = [r['_id'] for r in self[collection].find(filter_, {'_id': 1})]
            return _ids

    def _get_job_info(self, identifier):
        return self['_jobs'].find_one({'identifier': identifier})

    def get_meta_data(self, **kwargs):
        return self['_meta'].find_one(kwargs)['value']

    def _get_object_info(self, identifier, variety, **kwargs):
        return self['_objects'].find_one(
            {'identifier': identifier, 'variety': variety, **kwargs}
        )

    def _get_object_info_where(self, variety, **kwargs):
        return self['_objects'].find_one({'variety': variety, **kwargs})

    def _get_raw_cursor(self, *args, **kwargs):
        collection = self[args[1]]
        args = args[2:]
        return Cursor(collection, *args, **kwargs)

    def get_query_params_for_validation_set(self, validation_set):
        return ('find', '_validation_sets', {'identifier': validation_set})

    def _get_table_from_query_params(self, query_params):
        return query_params[1]

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

    def _modify_query_params_for_id_only(self, *args, **kwargs):
        method = args[0]
        collection = args[1]
        if len(args) >= 3:
            filter_ = args[2]
        else:
            filter_ = {}
        projection = {'_id': 1}
        return (method, collection, filter_, projection), kwargs

    def _modify_query_params_for_search(self, similar_ids, *args, **kwargs):
        method = args[0]
        collection = args[1]
        filter_ = {}
        projection = None
        if len(args) >= 3:
            filter_ = args[2]
        elif len(args) >= 4:
            projection = args[3]
        filter_ = {'_id': {'$in': similar_ids}, **filter_}
        return (method, collection, filter_, projection), kwargs

    def _query_is_trivial(self, *args, **kwargs):
        if len(args) <= 2:
            return True
        elif len(args) == 3:
            _, _, filter_ = args
        else:
            _, _, filter_, projection = args
        return bool(filter_)

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
        collection, filter_ = watcher_info['query_params']
        logging.info(f'unsetting neighbourhood {info["info"]}')
        return self[collection]._base_update(
            'update_many',
            filter_,
            {'$unset': {f'_like.{info["identifier"]}': 1}},
            refresh=False
        )

    def _unset_watcher_outputs(self, info):
        _, collection, filter_ = info['query_params']
        logging.info(f'unsetting output field _outputs.{info["key"]}.{info["model"]}')
        self[collection]._base_update(
            'update_many',
            filter_,
            {'$unset': {f'_outputs.{info["key"]}.{info["model"]}': 1}},
        )

    def _update_neighbourhood(self, ids, similar_ids, identifier, *query_params):
        collection = self[query_params[0]]
        collection.bulk_write(
            [
                UpdateOne({'_id': id_}, {'$set': {f'_like.{identifier}': sids}})
                for id_, sids in zip(ids, similar_ids)
            ]
        )

    def _update_job_info(self, identifier, key, value):
        self['_jobs'].update_one({'identifier': identifier}, {'$set': {key: value}})

    def _update_object_info(self, identifier, variety, key, value):
        self['_objects'].update_one(
            {'identifier': identifier, 'variety': variety}, {'$set': {key: value}}
        )

    def _update_to_select(self, method, collection, filter_, update, *args, **kwargs):
        return ('find', collection, filter_, *args), kwargs

    def write_output_to_job(self, identifier, msg, stream):
        assert stream in {'stdout', 'stderr'}
        self['_jobs'].update_one({'identifier': identifier}, {'$push': {stream: msg}})

    def _write_watcher_outputs(self, watcher_info, outputs, _ids):
        collection = watcher_info['query_params'][1]
        key = watcher_info.get('key', '_base')
        model_name = watcher_info['model']
        logging.info('bulk writing...')
        if watcher_info.get('target') is None:
            self[collection].bulk_write(
                [
                    UpdateOne(
                        {'_id': id},
                        {'$set': {f'_outputs.{key}.{model_name}': outputs[i]}},
                    )
                    for i, id in enumerate(_ids)
                ]
            )
        else:  # pragma: no cover
            self[collection].bulk_write(
                [
                    UpdateOne(
                        {'_id': id}, {'$set': {watcher_info['target']: outputs[i]}}
                    )
                    for i, id in enumerate(_ids)
                ]
            )
        logging.info('done.')

