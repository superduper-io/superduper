import random
import warnings
from typing import Any

from superduperdb.cluster.client import vector_search
from superduperdb.cluster.annotations import Tuple, CustomConvertible, List, \
    ObjectIdConvertible

warnings.filterwarnings('ignore')

from superduperdb.mongodb.cursor import SuperDuperCursor

from pymongo.collection import Collection as MongoCollection
from pymongo.cursor import Cursor
import torch.utils.data

from superduperdb.special_dicts import MongoStyleDict
from superduperdb.downloads import gather_urls, InMemoryDownloader
from superduperdb.models.utils import apply_model


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

        self.functions = self.database.functions
        self.measures = self.database.measures
        self.metrics = self.database.metrics
        self.models = self.database.models
        self.objectives = self.database.objectives
        self.preprocessors = self.database.preprocessors
        self.postprocessors = self.database.postprocessors
        self.splitters = self.database.splitters
        self.trainers = self.database.trainers
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

    def apply_model(self, *args, **kwargs):
        return self.database.apply_model(*args, **kwargs)

    def cancel_job(self, job_id):
        return self.database.cancel_job(job_id)

    @vector_search
    def clear_remote_cache(self):
        """
        Drop the hash_set currently in-use.
        """
        for k in self._all_hash_sets:
            del self._all_hash_sets[k]

    def create_agent(self, *args, **kwargs):
        return self.database.create_agent(*args, **kwargs)

    def create_function(self, *args, **kwargs):
        return self.database.create_function(*args, **kwargs)

    def create_imputation(self, *args, **kwargs):
        """
        Create imputation

        :param args: positional arguments to ``self.database.create_imputation``
        :param kwargs: passed to ``self.database.create_imputation``

        """
        return self.database.create_imputation(self.name, *args, **kwargs)

    def create_learning_task(self, *args, **kwargs):
        """
        Create learning task.

        :param args: positional arguments to ``self.database.create_learning_task``
        :param kwargs: passed to ``self.database.create_learning_task``
        """
        return self.database.create_learning_task(self.name, *args, **kwargs)

    def create_measure(self, *args, **kwargs):
        """
        Create measure.

        :param args: positional arguments to ``self.database.create_measure``
        :param kwargs: passed to ``self.database.create_measure``
        """
        return self.database.create_measure(*args, **kwargs)

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

    def create_objective(self, *args, **kwargs):
        """
        Create objective.

        :param args: positional arguments to ``self.database.create_objective``
        :param kwargs: passed to ``self.database.create_objective``
        """
        return self.database.create_objective(*args, **kwargs)

    def create_semantic_index(self, *args, **kwargs):
        """
        Create semantic-index.

        :param args: positional arguments to ``self.database.create_semantic_index``
        :param kwargs: passed to ``self.database.create_semantic_index``
        """
        return self.database.create_semantic_index(self.name,
                                                   *args,
                                                   **kwargs)

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

    def delete_measure(self, *args, **kwargs):
        """
        Delete measure
        """
        return self.database.delete_measure(*args, **kwargs)

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

    def delete_objective(self, *args, **kwargs):
        """
        Delete objective
        """
        return self.database.delete_objective(*args, **kwargs)

    def delete_semantic_index(self, *args, **kwargs):
        """
        Delete semantic index
        """
        return self.database.delete_semantic_index(*args, **kwargs)

    def delete_trainer(self, *args, **kwargs):
        """
        Delete trainer
        """
        return self.database.delete_trainer(*args, **kwargs)

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

    def list_agents(self):
        """
        List agents.
        """
        return self.database.list_agents()

    def list_functions(self):
        """
        List functions.
        """
        return self.database.list_functions()

    def list_imputations(self):
        """
        List imputations.
        """
        return self.database.list_imputations()

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

    def list_objectives(self):
        """
        List objectives.
        """
        return self.database.list_objectives()

    def list_preprocessors(self):
        """
        List preprocessors.
        """
        return self.database.preprocessors()

    def list_postprocessors(self):
        """
        List postprocessors.
        """
        return self.database.preprocessors()

    def list_semantic_indexes(self):
        """
        List semantic-indexes.
        """
        return self.database.list_semantic_indexes()

    def list_trainers(self):
        """
        List trainers.
        """
        return self.database.list_trainers()

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
        urls = gather_urls([filter], gather_ids=False)[0]
        if urls:
            filter = MongoStyleDict(filter)
            urls, keys, _ = gather_urls([filter], gather_ids=False)
            downloader = InMemoryDownloader(urls)
            downloader.go()
            for i, (k, url) in enumerate(zip(keys, urls)):
                filter[k]['_content']['bytes'] = downloader.results[i]
            filter = self.database.convert_from_bytes_to_types(filter)
        return filter

    def find(self, filter=None, *args, like=None, n=10, similar_first=False, raw=False,
             features=None, download=False, similar_join=None, semantic_index=None, **kwargs):
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
            if semantic_index is None:
                semantic_index = self.database.get_meta_data(key='semantic_index',
                                                             collection=self.name)
            if similar_first:
                return self._find_similar_then_matches(filter, like, *args, raw=raw,
                                                       features=features, like=like,
                                                       semantic_index=semantic_index,
                                                       n=n,
                                                       **kwargs)
            else:
                return self._find_matches_then_similar(filter, like, *args, raw=raw,
                                                       n=n,
                                                       features=features, semantic_index=semantic_index,
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
    def find_nearest(self, like: CustomConvertible(), ids=None, n=10, semantic_index=None,
                     remote=None) -> Tuple([Any, List(ObjectIdConvertible())]):
        hash_set = self.database._get_hash_set(semantic_index)
        if ids is not None:
            hash_set = hash_set[ids]

        if '_id' in like:
            return hash_set.find_nearest_from_id(like['_id'], n=n)
        else:
            if semantic_index is None:
                semantic_index = self.database._get_meta_data('semantic_index')
            si_info = self.database.get_object_info(semantic_index, 'learning_task',
                                                    task_type='semantic_index')
            models = si_info['models']
            keys = si_info['keys']
            available_keys = list(like.keys()) + ['_base']
            model, key = next((m, k) for m, k in zip(models, keys) if k in available_keys)
            document = MongoStyleDict(like)
            if '_outputs' not in document:
                document['_outputs'] = {}
            key_to_watch = si_info['keys_to_watch'][0]
            model_identifier = next(m for i, m in enumerate(si_info['models'])
                                    if si_info['keys'][i] == key_to_watch)
            watcher_info = self.database.get_object_info(f'[{semantic_index}]:{model_identifier}/{key_to_watch}',
                                                         'watcher')
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
            with torch.no_grad():
                h = apply_model(model, model_input, True)
        return hash_set.find_nearest_from_hash(h, n=n)

    def _find_similar_then_matches(self, filter, like, *args, raw=False, features=None, n=10,
                                   semantic_index=None,
                                   **kwargs):
        similar_ids, scores = self.find_nearest(like, n=n, semantic_index=semantic_index)
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
                                   semantic_index=None, **kwargs):
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
                self.find_nearest(like, ids=ids, n=n, semantic_index=semantic_index)
        else:  # pragma: no cover
            similar_ids, scores = \
                self.find_nearest(like, n=n, semantic_index=semantic_index)

        if raw:
            return Cursor(self, {'_id': {'$in': similar_ids}}, **kwargs)  # pragma: no cover
        else:
            return SuperDuperCursor(self, {'_id': {'$in': similar_ids}},
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