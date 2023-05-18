import functools
import random
import warnings
from typing import Any

from superduperdb.datalayer.base.database import BaseDatabase
from superduperdb.cluster.client_decorators import vector_search
from superduperdb.cluster.annotations import Tuple, List, \
    ObjectIdConvertible, Convertible

warnings.filterwarnings('ignore')

from superduperdb.datalayer.mongodb.cursor import SuperDuperCursor

from pymongo.collection import Collection as MongoCollection
from pymongo.cursor import Cursor
import torch.utils.data

from superduperdb.misc.special_dicts import MongoStyleDict
from superduperdb.fetchers.downloads import gather_uris, InMemoryDownloader
from superduperdb.models.torch.wrapper import apply_model


class Collection(MongoCollection):
    """
    Collection building on top of ``pymongo.collection.Collection``.
    Implements additional methods required by ``superduperdb`` for AI/ machine learning.
    """

    _id = '_id'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._hash_set = None
        self._semantic_index = None

        self.metrics = self.database.metrics
        self.models = self.database.models
        self.types = self.database.types

        self._all_hash_sets = self.database._all_hash_sets

    @property
    def remote(self):
        return self.database.remote

    @remote.setter
    def remote(self, value):
        self.database.remote = value

    def __getitem__(self, item):
        if item == '_validation_sets':
            return self.database[f'{self.name}._validation_sets']
        return super().__getitem__(item)

    @property
    def hash_set(self):
        return self.database._get_hash_set(self.database._get_meta_data(
            key='semantic_index',
            collection=self.name,
        )['value'])

    @property
    def semantic_index(self):
        if self._semantic_index is None:
            try:
                si = self['_meta'].find_one({'key': 'semantic_index'})['value']
                self._semantic_index = si
                return si
            except TypeError:  # pragma: no cover
                return
        return self._semantic_index

    @semantic_index.setter
    def semantic_index(self, value):
        self._semantic_index = value

    def predict(self, *args, **kwargs):
        return self.database.predict(*args, **kwargs)

    def predict_one(self, *args, **kwargs):
        return self.database.predict_one(*args, **kwargs)

    def cancel_job(self, job_id):
        return self.database.cancel_job(job_id)

    @vector_search
    def clear_remote_cache(self):
        """
        Drop the hash_set currently in-use.
        """
        for k in self._all_hash_sets:
            del self._all_hash_sets[k]

    @functools.wraps(BaseDatabase.create_learning_task)
    def create_learning_task(self, models, keys, *query_params, **kwargs):
        """
        Create learning task.

        :param args: positional arguments to ``self.database.create_learning_task``
        :param kwargs: passed to ``self.database.create_learning_task``
        """
        return self.database.create_learning_task(models, keys, *(self.name, *query_params),
                                                  **kwargs)

    def create_metric(self, *args, **kwargs):
        """
        Create metric.

        :param args: positional arguments to ``self.database.create_metric``
        :param kwargs: passed to ``self.database.create_metric``
        """
        return self.database.create_metric(*args, **kwargs)

    def create_model(self, *args, **kwargs):
        """
        Create a model.

        :param args: positional arguments to ``self.database.create_model``
        :param kwargs: passed to ``self.database.create_model``
        """
        return self.database.create_model(*args, **kwargs)

    def create_neighbourhood(self, *args, **kwargs):
        """
        Create neighbourhood.

        :param args: positional arguments to ``self.database.create_neighbourhood``
        :param kwargs: passed to ``self.database.create_neighbourhood``
        """
        return self.database.create_neighbourhood(*args, **kwargs)

    def create_trainer(self, *args, **kwargs):
        """
        Create trainer.

        :param args: positional arguments to ``self.database.create_trainer``
        :param kwargs: passed to ``self.database.create_trainer``
        """
        return self.database.create_trainer(*args, **kwargs)

    def create_type(self, *args, **kwargs):
        """
        Create type.

        :param args: positional arguments to ``self.database.create_type``
        :param kwargs: passed to ``self.database.create_type``
        """
        return self.database.create_type(*args, **kwargs)

    def create_validation_set(self, identifier, filter_=None, *args, **kwargs):
        """
        Create validation set.

        :param identifier: identifier of validation-set
        :param filter_: filter_ defining where to get data from
        :param args: positional arguments to ``self.database.create_validation_set``
        :param kwargs: passed to ``self.database.create_validation_set``
        """
        if filter_ is None:
            filter_ = {'_fold': 'valid'}
        else:
            filter_['_fold'] = 'valid'
        return self.database.create_validation_set(identifier, self.name, filter_, *args, **kwargs)

    def create_watcher(self, *args, **kwargs):
        """
        Create watcher.

        :param args: positional arguments to ``self.database.create_watcher``
        :param kwargs: passed to ``self.database.create_watcher``
        """
        return self.database.create_watcher(self.name, *args, **kwargs)

    def delete_agent(self, *args, **kwargs):
        """
        Delete agent
        """
        return self.database.delete_agent(*args, **kwargs)

    def delete_function(self, *args, **kwargs):
        """
        Delete function
        """
        return self.database.delete_function(*args, **kwargs)

    def delete_imputation(self, *args, **kwargs):
        """
        Delete imputation
        """
        return self.database.delete_imputation(*args, **kwargs)

    def delete_learning_task(self, *args, **kwargs):
        """
        Delete learning task
        """
        return self.database.delete_learning_task(*args, **kwargs)

    def delete_metric(self, *args, **kwargs):
        """
        Delete metric
        """
        return self.database.delete_metric(*args, **kwargs)

    def delete_model(self, *args, **kwargs):
        """
        Delete model
        """
        return self.database.delete_model(*args, **kwargs)

    def delete_neighbourhood(self, *args, **kwargs):
        """
        Delete neighbourhood
        """
        return self.database.delete_neighbourhood(*args, **kwargs)

    def delete_type(self, *args, **kwargs):
        """
        Delete type
        """
        return self.database.delete_type(*args, **kwargs)

    def delete_validation_set(self, *args, **kwargs):
        """
        Delete validation-set
        """
        return self.database.delete_validation_set(*args, **kwargs)

    def delete_watcher(self, *args, **kwargs):
        """
        Delete watcher
        """
        return self.database.delete_watcher(*args, **kwargs)

    def execute_query(self, filter_, projection=None):
        return self.find(filter_, projection=projection)

    def list_learning_tasks(self):
        """
        List learning-tasks.
        """
        return self.database.list_learning_tasks()

    def list_jobs(self):
        """
        List jobs.
        """
        return self.database.list_jobs()

    def list_metrics(self):
        """
        List metrics.
        """
        return self.database.list_metrics()

    def list_models(self):
        """
        List models.
        """
        return self.database.list_models()

    def list_types(self):
        """
        List types.
        """
        return self.database.list_types()

    def list_watchers(self):
        """
        List watchers.
        """
        return self.database.list_watchers()

    def _get_content_for_filter(self, filter):
        uris = gather_uris([filter], gather_ids=False)[0]
        if uris:
            filter = MongoStyleDict(filter)
            uris, keys, _ = gather_uris([filter], gather_ids=False)
            downloader = InMemoryDownloader(uris)
            downloader.go()
            for i, (k, uri) in enumerate(zip(keys, uris)):
                filter[k]['_content']['bytes'] = downloader.results[i]
            filter = self.database.convert_from_bytes_to_types(filter)
        return filter

    def find(self, filter=None, *args, like=None, n=10, similar_first=False, raw=False,
             features=None, download=False, similar_join=None, semantic_index=None,
             watcher=None, hash_set_cls='vanilla', measure='css', **kwargs):
        """
        Behaves like MongoDB ``find`` with similarity search as additional option.

        :param filter: filter dictionary
        :param args: args passed to super()
        :param like: item to which the results of the find should be similar
        :param similar_first: toggle to ``True`` to first find similar things, and then
                              apply filter to these things
        :param raw: toggle to ``True`` to not convert bytes to Python objects but return raw bytes
        :param features: dictionary of model outputs to replace for dictionary elements
        :param download: toggle to ``True`` in case query has downloadable "_content" components
        :param similar_join: replace ids by documents
        :param kwargs: kwargs to be passed to super()
        """
        if filter is None:
            filter = {}
        if download and like is not None:
            like = self._get_content_for_filter(like)    # pragma: no cover
        if like is not None:
            if similar_first:
                return self._find_similar_then_matches(filter, like, *args, raw=raw,
                                                       features=features, like=like,
                                                       semantic_index=semantic_index,
                                                       n=n,
                                                       measure=measure,
                                                       watcher=watcher,
                                                       hash_set_cls=hash_set_cls,
                                                       **kwargs)
            else:
                return self._find_matches_then_similar(filter, like, *args, raw=raw,
                                                       n=n,
                                                       features=features,
                                                       measure=measure,
                                                       semantic_index=semantic_index,
                                                       hash_set_cls=hash_set_cls,
                                                       watcher=watcher,
                                                       **kwargs)
        else:
            if raw:
                return Cursor(self, filter, *args, **kwargs)
            else:
                return SuperDuperCursor(self, filter, *args, features=features,
                                        similar_join=similar_join, **kwargs)

    def find_one(self, *args, **kwargs):
        """
        Behaves like MongoDB ``find_one`` with exception of ``$like`` operator.
        See *Collection.find* for more details.

        :param args: args passed to super()
        :param kwargs: kwargs to be passed to super()
        """
        cursor = self.find(*args, **kwargs)
        for result in cursor.limit(-1):
            return result

    def insert_one(self, document, *args, **kwargs):
        """
        Insert a document into database.

        :param document: list of documents
        :param args: args to be passed to super()
        :param kwargs: kwargs to be passed to super()
        """
        return self.insert_many([document], *args, **kwargs)

    def insert_many(self, documents, *args, verbose=True, refresh=True, **kwargs):
        """
        Insert many documents into database.

        :param documents: list of documents
        :param args: args to be passed to super()
        :param verbose: toggle to ``True`` to display outputs during computation
        :param refresh: toggle to ``False`` to suppress model processing
        :param kwargs: kwargs to be passed to super()
        """
        for document in documents:
            r = random.random()
            try:
                valid_probability = self['_meta'].find_one({'key': 'valid_probability'})['value']
            except TypeError:
                valid_probability = 0.05
            if '_fold' not in document:
                document['_fold'] = 'valid' if r < valid_probability else 'train'
        documents = self._infer_types(documents)

        output = super().insert_many(documents, *args, **kwargs)
        if not refresh:  # pragma: no cover
            return output, None
        task_graph = self.database._build_task_workflow((self.name,),
                                                        ids=output.inserted_ids,
                                                        verbose=verbose)
        task_graph()
        return output, task_graph

    def refresh_watcher(self, *args, **kwargs):
        """
        Recompute model outputs.

        :param args: position args passed to ``self.database.refresh_watcher``
        :param kwargs: kwargs passed to ``self.database.refresh_watcher``
        """
        return self.database.refresh_watcher(self.name, *args, **kwargs)

    def replace_one(self, filter, replacement, *args, refresh=True, **kwargs):
        """
        Replace a document in the database. The outputs of models will be refreshed for this
        document.

        :param filter: MongoDB like filter
        :param replacement: Replacement document
        :param args: args to be passed to super()
        :param refresh: Toggle to *False* to not process document again with models.
        :param kwargs: kwargs to be passed to super()
        """
        id_ = super().find_one(filter, *args, **kwargs)['_id']
        replacement = self.convert_from_types_to_bytes(replacement)
        result = super().replace_one({'_id': id_}, replacement, *args, **kwargs)
        if refresh and self.list_models():
            self._process_documents([id_])
        return result

    def convert_from_types_to_bytes(self, r):
        """
        Convert non MongoDB types to bytes using types already registered with collection.

        :param r: record in which to convert types
        :return modified record
        """
        return self.database.convert_from_types_to_bytes(r)

    def update_many(self, filter, *args, refresh=True, **kwargs):
        """
        Update the collection at the documents specified by the filter. If there are active
        models these are applied to the updated documents.

        :param filter: Filter dictionary selecting documents to be updated
        :param args: Arguments to be passed to ``super()``
        :param refresh: Toggle to ``False`` to stop models being applied to updated documents
        :param kwargs: Keyword arguments to be passed to ``super()``
        :return: ``result`` or ``(result, job_ids)`` depending on ``self.remote``
        """
        if refresh and self.list_models():
            ids = [r['_id'] for r in super().find(filter, {'_id': 1})]
        args = list(args)
        args[0] = self.convert_from_types_to_bytes(args[0])
        args = tuple(args)
        result = super().update_many(filter, *args, **kwargs)
        if refresh and self.list_models():
            job_ids = self.database._process_documents(self.name, ids=ids)
            return result, job_ids
        return result

    def update_one(self, filter, *args, refresh=True, **kwargs):
        """
        Update a single document specified by the filter.

        :param filter: Filter dictionary selecting documents to be updated
        :param args: Arguments to be passed to ``super()``
        :param refresh: Toggle to ``False`` to stop models being applied to updated documents
        :param kwargs: Keyword arguments to be passed to ``super()``
        :return: ``result`` or ``(result, job_ids)`` depending on ``self.remote``
        """
        id_ = super().find_one(filter, {'_id': 1})['_id']
        return self.update_many({'_id': id_}, *args, refresh=refresh, **kwargs)

    @vector_search
    def find_nearest(self, like: Convertible(), ids=None, n=10,
                     watcher=None,
                     semantic_index=None,
                     measure='css',
                     hash_set_cls='vanilla') -> Tuple([List(ObjectIdConvertible()), Any]):

        if semantic_index is not None:
            si_info = self.database.get_object_info(semantic_index, variety='learning_task')
            models = si_info['models']
            keys = si_info['keys']
            watcher = self.database._get_watcher_for_learning_task(semantic_index)
            watcher_info = self.database.get_object_info(watcher, variety='watcher')
        else:
            watcher_info = self.database.get_object_info(watcher, variety='watcher')
            models = [watcher_info['model']]
            keys = [watcher_info['key']]

        if watcher not in self.database._all_hash_sets:
            self.database._load_hashes(watcher=watcher, measure=measure,
                                       hash_set_cls=hash_set_cls)

        hash_set = self.database._all_hash_sets[watcher]
        if ids is not None:
            hash_set = hash_set[ids]

        if '_id' in like:
            return hash_set.find_nearest_from_id(like['_id'], n=n)

        available_keys = list(like.keys()) + ['_base']
        model, key = next((m, k) for m, k in zip(models, keys) if k in available_keys)
        document = MongoStyleDict(like)
        if '_outputs' not in document:
            document['_outputs'] = {}
        features = watcher_info.get('features', {})
        for subkey in features:
            if subkey not in document:
                continue
            if subkey not in document['_outputs']:
                document['_outputs'][subkey] = {}
            if features[subkey] not in document['_outputs'][subkey]:
                document['_outputs'][subkey][features[subkey]] = \
                    apply_model(self.models[features[subkey]], document[subkey])
            document[subkey] = document['_outputs'][subkey][features[subkey]]
        model_input = document[key] if key != '_base' else document
        model = self.models[model]
        h = model.predict_one(model_input)
        return hash_set.find_nearest_from_hash(h, n=n)

    def _find_similar_then_matches(self, filter, like, *args, raw=False, features=None, n=10,
                                   semantic_index=None, watcher=None, hash_set_cls='vanilla',
                                   **kwargs):
        similar_ids, scores = self.find_nearest(like, n=n, semantic_index=semantic_index,
                                                watcher=watcher, hash_set_cls='vanilla')
        filter = {
            '$and': [
                filter,
                {'_id': {'$in': similar_ids}}
            ]
        }
        if raw:
            return Cursor(self, filter, *args, **kwargs)  # pragma: no cover
        else:
            return SuperDuperCursor(
                self,
                filter,
                *args,
                features=features,
                scores=dict(zip(similar_ids, scores)),
                **kwargs,
            )

    def _find_matches_then_similar(self, filter, like, *args, raw=False, features=None, n=10,
                                   semantic_index=None, watcher=None, hash_set_cls='vanilla',
                                   measure='css', **kwargs):
        if filter:
            matches_cursor = SuperDuperCursor(
                self,
                filter,
                {'_id': 1},
                *args[1:],
                features=features,
                **kwargs,
            )
            ids = [x['_id'] for x in matches_cursor]
            similar_ids, scores = \
                self.find_nearest(like, ids=ids, n=n, semantic_index=semantic_index,
                                  watcher=watcher, hash_set_cls=hash_set_cls,
                                  measure=measure)
        else:  # pragma: no cover
            similar_ids, scores = \
                self.find_nearest(like, n=n, semantic_index=semantic_index,
                                  watcher=watcher, hash_set_cls=hash_set_cls,
                                  measure=measure)

        if raw:
            return Cursor(self, {'_id': {'$in': similar_ids}}, *args, **kwargs)  # pragma: no cover
        else:
            return SuperDuperCursor(self, {'_id': {'$in': similar_ids}}, *args,
                                    features=features,
                                    scores=dict(zip(similar_ids, scores)), **kwargs)

    def _infer_types(self, documents):
        for r in documents:
            self.convert_from_types_to_bytes(r)
        return documents

    def watch_job(self, *args, **kwargs):
        """
        Watch stdout/stderr of worker job.
        """
        return self.database.watch_job(*args, **kwargs)

    def _write_watcher_outputs(self, watcher_info, outputs, _ids):
        return self.database._write_watcher_outputs(
            {**watcher_info, 'query_params': (self.name, *watcher_info['query_params'])},
            outputs,
            _ids,
        )