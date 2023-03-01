import os
import time

import click
from collections import defaultdict
import multiprocessing

import gridfs
import math
import networkx
import random
import warnings

from bson import ObjectId
from numpy.random import permutation

warnings.filterwarnings('ignore')

from superduperdb.cursor import SuperDuperCursor
from superduperdb.training.validation import validate_representations
from superduperdb.types.utils import convert_types

from pymongo import UpdateOne
from pymongo.collection import Collection as BaseCollection
from pymongo.cursor import Cursor
import torch.utils.data

from superduperdb import cf
from superduperdb.lookup import hashes
from superduperdb.models import loading
from superduperdb import getters as superduper_requests
from superduperdb import training
from superduperdb.utils import unpack_batch, MongoStyleDict, Downloader, progressbar, \
    ArgumentDefaultDict
from superduperdb.models.utils import apply_model, create_container


class Collection(BaseCollection):
    """
    SuperDuperDB collection type, subclassing *pymongo.collection.Collection*.
    Key methods are:

    Creating objects:

    - ``create_object``
    - ``create_model``
    - ``create_watcher``
    - ``create_neighbourhood``
    - ``create_semantic_index``

    Deleting objects:

    - ``delete_object``

    Accessing data:

    - ``find_one``
    - ``find``

    Inserting and updating data:
    - ``insert_many``
    - ``insert_one``
    - ``replace_one``
    - ``update_one``
    - ``update_many``

    Viewing meta-data

    - ``list_objects``

    Watching jobs

    - ``watch_job``

    Key properties:

    - ``hash_set`` (in memory vectors for neighbourhood search)
    - ``objects``

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._hash_set = None
        self._all_hash_sets = ArgumentDefaultDict(self._load_hashes)
        self.remote = cf.get('remote', False)
        self._filesystem = None
        self._filesystem_name = f'_{self.database.name}:{self.name}:files'
        self._semantic_index = None
        self._type_lookup = None

        self.objects = ArgumentDefaultDict(
            lambda x: ArgumentDefaultDict(lambda y: self._load_object(x[:-1], y))
        )
        self.models = ArgumentDefaultDict(lambda x: self._load_model(x))
        self._allowed_varieties = ['model', 'function', 'preprocessor', 'postprocessor',
                                   'splitter', 'objective', 'measure', 'type', 'metric']
        self.headers = None

    def __getitem__(self, item):
        if item == '_validation_sets':
            return self.database[f'{self.name}._validation_sets']
        return super().__getitem__(item)

    @property
    def filesystem(self):
        if self._filesystem is None:
            self._filesystem = gridfs.GridFS(
                self.database.client[self._filesystem_name]
            )
        return self._filesystem

    @property
    def hash_set(self):
        if self.semantic_index is None:
            raise Exception('No semantic index has been set!')  # pragma: no cover
        return self._all_hash_sets[self.semantic_index]

    @property
    def parent_if_appl(self):
        collection = self
        if self.name.endswith('_validation_sets'):
            parent = '.'.join(self.name.split('.')[:-1])
            collection = self.database[parent]
        return collection

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

    @property
    def type_lookup(self):
        if self._type_lookup is None:
            self._type_lookup = {}
            for t in self.list_types():
                try:
                    for s in self.types[t].types:
                        self._type_lookup[s] = t
                except AttributeError:
                    continue
        return self._type_lookup

    def apply_model(self, name, input_, **kwargs):
        if self.remote:
            return superduper_requests.client.apply_model(name, input_, **kwargs)
        return apply_model(self.models[name], input_, **kwargs)

    def create_imputation(self, name, model, model_key, target, target_key,
                          objective=None, metrics=None,
                          splitter=None, watch=True, **trainer_kwargs):
        """
        Create an imputation setup. This is any learning task where we have an input to the model
        compared to the target.

        :param name: Name of imputation
        :param model: Model settings or model name
        :param model_key: Key for model to injest
        :param target: Target settings (input to ``create_model``) or target name
        :param target_key: Key for model to predict
        :param objective: Loss settings or objective
        :param metrics: List of metric settings or metric names
        :param splitter: Splitter name to use to prepare data points for insertion to model
        :param trainer_kwargs: Keyword-arguments to forward to ``train_tools.ImputationTrainer``
        :return: job_ids of jobs required to create the imputation
        """

        assert name not in self.list_imputations()
        assert target in self.list_functions()
        assert model in self.list_models()
        if objective is not None:
            assert objective in self.list_objectives()
        if metrics:
            for metric in metrics:
                assert metric in self.list_metrics()

        self['_objects'].insert_one({
            'varieties': 'imputation',
            'name': name,
            'model': model,
            'model_key': model_key,
            'target': target,
            'target_key': target_key,
            'metrics': metrics or [],
            'objective': objective,
            'splitter': splitter,
            'trainer_kwargs': trainer_kwargs,
        })

        if objective is None:
            return

        try:
            jobs = [self._train_imputation(name)]
        except KeyboardInterrupt:
            print('aborting training early...')
            jobs = []
        if watch:
            jobs.append(self.create_watcher(
                model=model,
                key=model_key,
                filter_=trainer_kwargs.get('filter', {}),
                features=trainer_kwargs.get('features', {}),
                dependencies=jobs,
                verbose=True,
            ))
        return jobs

    def create_watcher(self, model, key='_base', filter_=None, verbose=False, target=None,
                       process_docs=True, features=None, loader_kwargs=None,
                       dependencies=()):

        assert self['_objects'].count_documents({'model': model, 'key': key,
                                                 'varieties': 'watcher'}) == 0, \
            f"This watcher {model}, {key} already exists"

        self['_objects'].insert_one({
            'varieties': ['watcher'],
            'model': model,
            'filter': filter_ if filter_ else {},
            'key': key,
            'features': features if features else {},
            'target': target,
            'loader_kwargs': loader_kwargs or {},
        })

        if process_docs:
            ids = [r['_id'] for r in self.find(filter_ if filter_ else {}, {'_id': 1})]
            if not ids:
                return
            if not self.remote:
                self._process_documents_with_watcher(model, key, ids, verbose=verbose)
            else:  # pragma: no cover
                return superduper_requests.jobs.process(
                    self.database.name,
                    self.name,
                    '_process_documents_with_watcher',
                    model,
                    key,
                    ids,
                    verbose=verbose,
                    dependencies=dependencies,
                )

    def create_model(self, name, object=None, preprocessor=None,
                     postprocessor=None, type=None):
        """
        Create a model registered in the collection directly from a python session.
        The added model will then watch incoming records and add outputs computed on those
        records into the ``"_outputs"`` fields of the records.
        The model is then stored inside MongoDB and can be accessed using the ``SuperDuperClient``.

        :param name: name of model
        :param object: if specified the model object (pickle-able) else None if model already exists
        :param preprocessor: separate preprocessing
        :param forward: separate forward pass
        :param postprocessor: separate postprocessing
        :param type: type for converting model outputs back and forth from bytes
        """
        assert name not in self['_objects'].distinct('name', {'varieties': 'model'}), \
            f'Model {name} already exists!'

        assert name not in self['_objects'].distinct('name', {'varieties': 'function'}), \
            f'Function {name} already exists!'

        if type is not None:
            assert type in self['_objects'].distinct('name', {'varieties': 'type'})

        file_id = self._create_pickled_file(object)

        self['_objects'].insert_one({
            'varieties': ['model', 'function'],
            'name': name,
            'object': file_id,
            'type': type,
            'preprocessor': preprocessor,
            'postprocessor': postprocessor,
        })

    def create_neighbourhood(self, name, n=10, watcher=None, batch_size=100):
        assert name not in self.list_neighbourhoods()
        self['_objects'].insert_one({
            'name': name,
            'watcher': watcher,
            'n': n,
            'batch_size': batch_size,
            'varieties': 'neighbourhood',
        })
        info = self['_objects'].find_one({'name': watcher})
        watcher_info = list(self['_objects'].find({'model': {'$in': info['models']},
                                                   'varieties': 'watcher'}))[0]
        filter_ = watcher_info.get('filter', {})
        ids = [r['_id'] for r in self.find(filter_, {'_id': 1})]
        if not self.remote:
            self._compute_neighbourhood(name, ids)
        else:
            return superduper_requests.jobs.process(
                self.database.name,
                self.name,
                '_compute_neighbourhood',
                name,
                ids=ids,
            )

    def create_semantic_index(self, name, models, measure, keys, validation_sets=(), metrics=(),
                              objective=None, filter_=None, splitter=None, loader_kwargs=None,
                              **trainer_kwargs):
        """
        :param name: Name of index
        :param models: List of existing models
        :param measure: Measure name
        :param keys: Keys in incoming data to listen to
        :param validation_sets: Name of immutable validation set to be used to evaluate performance
        :param metrics: List of existing metrics,
        :param objective: Loss name
        :param splitter: Splitter name
        :param filter: Filter on which to train
        :param trainer_kwargs: Keyword arguments to be passed to ``training.train_tools.RepresentationTrainer``
        :return: List of job identifiers if ``self.remote``
        """
        assert name not in self.list_semantic_indexes()

        if objective is not None:  # pragma: no cover
            if len(models) == 1:
                assert splitter is not None, 'need a splitter for self-supervised ranking...'

        self['_objects'].insert_one({
            'varieties': ['semantic_index'],
            'name': name,
            'models': models,
            'keys': keys,
            'metrics': metrics,
            'objective': objective,
            'measure': measure,
            'splitter': splitter,
            'filter': filter_ or {},
            'validation_sets': list(validation_sets),
            'trainer_kwargs': trainer_kwargs,
        })
        self.create_watcher(models[0], keys[0], filter_=filter_, process_docs=False,
                            features=trainer_kwargs.get('features', {}),
                            loader_kwargs=loader_kwargs or {})
        if objective is None:
            return [self.refresh_watcher(models[0], keys[0], dependencies=())]
        try:
            jobs = [self._train_semantic_index(name=name)]
        except KeyboardInterrupt:
            print('training aborted...')
            jobs = []
        jobs.append(self.refresh_watcher(models[0], keys[0], dependencies=jobs))
        return jobs

    def create_validation_set(self, name, filter=None, chunk_size=1000, splitter=None,
                              sample_size=None):
        """
        :param filter: filter (not including {"_fold": "valid"}
        """
        filter = filter or {}
        existing = self.list_validation_sets()
        if name in existing:
            raise Exception(f'validation set {name} already exists!')
        if isinstance(splitter, str):
            splitter = self.splitters[splitter]
        filter['_fold'] = 'valid'
        if sample_size is None:
            cursor = self.find(filter, {'_id': 0}, raw=True)
            total = self.count_documents(filter)
        else:
            assert isinstance(sample_size, int)
            _ids = self.distinct('_id', filter)
            if len(_ids) > sample_size:
                _ids = [_ids[int(i)] for i in permutation(sample_size)]
            cursor = self.find({'_id': {'$in': _ids}}, {'_id': 0}, raw=True)
            total = sample_size
        it = 0
        tmp = []
        for r in progressbar(cursor, total=total):
            if splitter is not None:
                r, other = splitter(r)
                r['_other'] = other
                r['_fold'] = 'valid'
            r['_validation_set'] = name
            tmp.append(r)
            it += 1
            if it % chunk_size == 0:
                self['_validation_sets'].insert_many(tmp)
                tmp = []
        if tmp:
            self['_validation_sets'].insert_many(tmp)

    def delete_imputation(self, name, force=False):
        """
        Delete imputation from collection

        :param name: Name of imputation
        :param force: Toggle to ``True`` to skip confirmation
        """
        do_delete = False
        if force or click.confirm(f'Are you sure you want to delete the imputation "{name}"?',
                                  default=False):
            do_delete = True
        if not do_delete:
            return

        info = self['_objects'].find_one({'name': name, 'varieties': 'imputation'})
        self['_objects'].delete_one({'model': info['model'], 'key': info['model_key'],
                                     'varieties': 'watcher'})
        self['_objects'].delete_one({'name': name, 'varieties': 'imputation'})

    def delete_watcher(self, model, key, force=False, delete_outputs=True):
        """
        Delete model from collection

        :param name: Name of model
        :param force: Toggle to ``True`` to skip confirmation
        """
        info = self['_objects'].find_one({'model': model, 'key': key,
                                          'varieties': 'watcher'})
        if not force: # pragma: no cover
            n_documents = self.count_documents(info.get('filter') or {})
        do_delete = False
        if force or click.confirm(f'Are you sure you want to delete this watcher: {model}, {key}; '
                                  f'{n_documents} documents will be affected.',
                                  default=False):
            do_delete = True
        if not do_delete:
            return

        if info.get('target') is None and delete_outputs:
            print(f'unsetting output field _outputs.{info["key"]}.{info["model"]}')
            super().update_many(
                info.get('filter') or {},
                {'$unset': {f'_outputs.{info["key"]}.{info["model"]}': 1}}
            )
        return self['_objects'].delete_one({'model': model, 'key': key,
                                            'varieties': 'watcher'})

    def delete_neighbourhood(self, name, force=False):
        """
        Delete neighbourhood from collection documents.

        :param name: Name of neighbourhood
        :param force: Toggle to ``True`` to skip confirmation
        """
        info = self['_objects'].find_one({'name': name, 'varieties': 'neighbourhood'})
        watcher_info = self['_objects'].find_one({'name': info['watcher']})
        filter_ = watcher_info['filter']
        n_documents = self.count_documents(filter_)
        if force or click.confirm(f'Removing neighbourhood "{name}", this will affect {n_documents}'
                                  ' documents. Are you sure?', default=False):
            self['_objects'].delete_one({'name': name})
            self.update_many(filter_, {'$unset': {f'_like.{name}': 1}}, refresh=False)
        else:
            print('aborting') # pragma: no cover

    def delete_semantic_index(self, name, force=False):
        """
        Delete semantic index.

        :param name: Name of semantic index
        :param force: Toggle to ``True`` to skip confirmation
        """
        info = self['_objects'].find_one({'name': name, 'varieties': 'semantic_index'})
        watcher = self['_objects'].find_one({'model': info['models'][0], 'key': info['keys'][0]})
        if info is None:  # pragma: no cover
            return
        do_delete = False
        if force or \
                click.confirm(f'Are you sure you want to delete this semantic index: "{name}"; '):
            do_delete = True

        if not do_delete:
            return

        if watcher:
            self['_objects'].delete_one({'_id': watcher['_id']})
        self['_objects'].delete_one({'name': name, 'varieties': 'semantic_index'})

    def _get_content_for_filter(self, filter):
        if '_id' not in filter:
            filter['_id'] = 0
        urls = self._gather_urls([filter])[0]
        if urls:
            filter = self._download_content(documents=[filter],
                                            timeout=None, raises=True)[0]
            filter = convert_types(filter, converters=self.types)
        return filter

    def find(self, filter=None, *args, like=None, similar_first=False, raw=False,
             features=None, download=False, similar_join=None, **kwargs):
        """
        Behaves like MongoDB ``find`` with exception of ``$like`` operator.

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
            if '_id' in like:
                del like['_id']
        if like is not None:
            if similar_first:
                return self._find_similar_then_matches(filter, like, *args, raw=raw,
                                                       features=features, like=like, **kwargs)
            else:
                return self._find_matches_then_similar(filter, like, *args, raw=raw,
                                                       features=features, **kwargs)
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

        :param documents: list of documents
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
            return output
        job_ids = self._process_documents(output.inserted_ids, verbose=verbose)
        if not self.remote:
            return output
        return output, job_ids

    def list_objects(self, variety=None, query=None):
        if query is None:
            query = {}
        if variety is not None:
            return self.parent_if_appl['_objects'].distinct('name', {'varieties': variety, **query})
        else:
            return list(self.parent_if_appl['_objects'].find(query, {'name': 1, 'varieties': 1, '_id': 0}))

    def list_semantic_indexes(self, query=None):
        query = query or {}
        return [r['name'] for r in self['_objects'].find({**query, 'varieties': 'semantic_index'},
                                                          {'name': 1})]

    def list_watchers(self, query=None):
        if query is None:
            query = {}
        items = self.parent_if_appl['_objects'].find({**query, 'varieties': 'watcher'},
                                                     {'model': 1, 'key': 1, '_id': 0})
        return [(r['model'], r['key']) for r in items]

    def list_validation_sets(self):
        """
        List validation sets
        :return: list of validation sets
        """
        return self.parent_if_appl['_validation_sets'].distinct('_validation_set')

    def refresh_watcher(self, model, key, dependencies=()):
        """
        Recompute model outputs.

        :param model: Name of model.
        """
        info = self.parent_if_appl['_objects'].find_one(
            {'model': model, 'key': key, 'varieties': 'watcher'}
        )
        ids = self.distinct('_id', info.get('filter') or {})
        return self._submit_process_documents_with_watcher(model, key, sub_ids=ids,
                                                           dependencies=dependencies)

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
        replacement = self._convert_types(replacement)
        result = super().replace_one({'_id': id_}, replacement, *args, **kwargs)
        if refresh and self.list_models():
            self._process_documents([id_])
        return result

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
        args[0] = self._convert_types(args[0])
        args = tuple(args)
        result = super().update_many(filter, *args, **kwargs)
        if refresh and self.list_models():
            job_ids = self._process_documents(ids)
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

    def unset_hash_set(self):
        """
        Drop the hash_set currently in-use.
        """
        if self.semantic_index in self._all_hash_sets:
            del self._all_hash_sets[self.semantic_index]

    def watch_job(self, identifier):
        """
        Watch the standard output of a collection job.

        :param identifier: Job identifier
        """
        try:
            status = 'pending'
            n_lines = 0
            while status in {'pending', 'running'}:
                r = self['_jobs'].find_one(
                    {'identifier': identifier},
                    {'stdout': 1, 'status': 1},
                )
                status = r['status']
                if status == 'running':
                    if len(r['stdout']) > n_lines:
                        print(''.join(r['stdout'][n_lines:]), end='')
                    n_lines = len(r['stdout'])
                    time.sleep(0.2)
                else:
                    time.sleep(0.2)
            if status == 'success':
                r = self['_jobs'].find_one({'identifier': identifier}, {'stdout': 1, 'status': 1})
                if len(r['stdout']) > n_lines:
                    print(''.join(r['stdout'][n_lines:]), end='')
            elif status == 'failed': # pragma: no cover
                r = self['_jobs'].find_one({'identifier': identifier}, {'msg': 1})
                print(r['msg'])
        except KeyboardInterrupt: # pragma: no cover
            return

    def validate_semantic_index(self, name, validation_sets, metrics):
        results = {}
        features = self['_objects'].find_one({'name': name,
                                              'varieties': 'semantic_index'}).get('features')
        for vs in validation_sets:
            results[vs] = validate_representations(self, vs, name, metrics, features=features)
        for vs in results:
            for m in results[vs]:
                self['_objects'].update_one(
                    {'name': name, 'varieties': 'semantic_index'},
                    {'$set': {f'final_metrics.{vs}.{m}': results[vs][m]}}
                )

    def _compute_neighbourhood(self, name, ids):

        import sys
        sys.path.insert(0, os.getcwd())

        info = self['_objects'].find_one({'name': name, 'varieties': 'neighbourhood'})
        print('getting hash set')
        h = self._all_hash_sets[info['semantic_index']]
        print(h.shape)
        print(f'computing neighbours based on neighbour "{name}" and '
              f'index "{info["semantic_index"]}"')

        for i in progressbar(range(0, len(ids), info['batch_size'])):
            sub = ids[i: i + info['batch_size']]
            results = h.find_nearest_from_ids(sub, n=info['n'])
            similar_ids = [res['_ids'] for res in results]
            self.bulk_write([
                UpdateOne({'_id': id_}, {'$set': {f'_like.{name}': sids}})
                for id_, sids in zip(sub, similar_ids)
            ])

    def _create_plan(self):
        G = networkx.DiGraph()
        for watcher in self.list_watchers():
            G.add_node(('watcher', watcher))
        for model, key in self.list_watchers():
            deps = self._get_dependencies_for_watcher(model, key)
            for dep in deps:
                G.add_edge(('watcher', dep), ('watcher', (model, key)))
        for neigh in self.list_neighbourhoods():
            model, key = self._get_watcher_for_neighbourhood(neigh)
            G.add_edge(('watcher', (model, key)), ('neighbourhood', neigh))
        assert networkx.is_directed_acyclic_graph(G)
        return G

    def create_object(self, name, object, varieties):
        for variety in varieties:
            assert variety in self._allowed_varieties, \
                f'type "{variety}" not admissible; allowed types are {self._allowed_varieties}'
            assert self['_objects'].count_documents({'name': name, 'varieties': variety}) == 0, \
                f'An object with the name: "{name}" and variety "{variety}" already exists...'
        file_id = self._create_pickled_file(object)
        self[f'_objects'].insert_one({'name': name, 'object': file_id, 'varieties': varieties})

    def __getattr__(self, item):
        if item.startswith('create_'):
            variety = item.split('_')[-1]
            def create_local(name, object):
                return self.create_object(name, object, [variety])
            return create_local
        if item.startswith('delete_'):
            variety = item.split('_')[-1]
            def delete_local(name, force=False):
                assert self['_objects'].find_one({'name': name, 'varieties': variety})['varieties'] == [variety], \
                    'can\'t delete object in this way, since used may multiple types of objects'
                return self.delete_object([variety], name, force=force)
            return delete_local
        if item.startswith('list'):
            variety = item.split('_')[-1][:-1]
            def list_local(query=None):
                return self.list_objects(variety, query)
            return list_local
        if item[:-1] in self._allowed_varieties and item[-1] == 's':
            return self.objects[item]
        raise AttributeError(item)

    def _create_pickled_file(self, object):
        return loading.save(object, filesystem=self.filesystem)

    def _convert_types(self, r):
        """
        Convert non MongoDB types to bytes using types already registered with collection.

        :param r: record in which to convert types
        :return modified record
        """
        for k in r:
            if isinstance(r[k], dict) and '_content' not in r[k]:
                r[k] = self._convert_types(r[k])
            try:
                d = self.type_lookup
                t = d[type(r[k])]
            except KeyError:
                t = None
            if t is not None:
                r[k] = {'_content': {'bytes': self.types[t].encode(r[k]), 'type': t}}
        return r

    def delete_model(self, name, force=False):
        return self.delete_object(['model', 'function'], name, force=force)

    def delete_object(self, varieties, object, force=False):
        data = list(self[f'_objects'].find({'name': object, 'varieties': varieties}))
        if not data and not force:
            raise Exception(f'This object does not exist...{object}')
        for variety in varieties:
            if object in getattr(self, variety + 's'):
                del getattr(self, variety + 's')[object]

        if force or click.confirm(f'You are about to delete these {varieties}: {object}, are you sure?',
                                  default=False):
            r = self[f'_objects'].find_one({'name': object, 'varieties': varieties})
            self.filesystem.delete(r['object'])
            self[f'_objects'].delete_one({'name': r['name'], 'varieties': varieties})
        return data

    @staticmethod
    def _dict_to_str(d):
        sd = Collection._standardize_dict(d)
        return str(sd)

    def _download_content(self, ids=None, documents=None, timeout=None, raises=True,
                          n_download_workers=None, headers=None):
        update_db = False
        if documents is None:
            update_db = True
            assert ids is not None
            documents = list(self.find({'_id': {'$in': ids}}, {'_outputs': 0}, raw=True))
        urls, keys, place_ids = self._gather_urls(documents)
        print(f'found {len(urls)} urls')
        if not urls:
            return

        if n_download_workers is None:
            try:
                n_download_workers = self['_meta'].find_one({'key': 'n_download_workers'})['value']
            except TypeError:
                n_download_workers = 0

        if headers is None:
            try:
                headers = self['_meta'].find_one({'key': 'headers'})['value']
            except TypeError:
                headers = 0

        if timeout is None:
            try:
                timeout = self['_meta'].find_one({'key': 'download_timeout'})['value']
            except TypeError:
                timeout = None

        downloader = Downloader(
            urls=urls,
            ids=place_ids,
            keys=keys,
            collection=self,
            n_workers=n_download_workers,
            timeout=timeout,
            headers=headers,
            raises=raises,
            update_db=update_db,
        )
        downloader.go()
        if update_db:
            return
        for id_, key in zip(place_ids, keys):
            if not isinstance(documents[id_], MongoStyleDict):
                documents[id_] = MongoStyleDict(documents[id_])
            documents[id_][f'{key}._content.bytes'] = downloader.results[id_]
        return documents

    def _find_nearest(self, like, ids=None, n=10):
        if self.remote:
            like = self._convert_types(like)
            return superduper_requests.client._find_nearest(self.database.name, self.name, like,
                                                            ids=ids)
        if ids is None:
            hash_set = self.hash_set
        else:  # pragma: no cover
            hash_set = self.hash_set[ids]

        if '_id' in like:
            return hash_set.find_nearest_from_id(like['_id'],
                                                 n=n)
        else:
            si_info = self.parent_if_appl['_objects'].find_one(
                {'name': self.semantic_index, 'varieties': 'semantic_index'}
            )
            models = si_info['models']
            keys = si_info['keys']
            watcher_model, watcher_key = (models[0], keys[0])
            available_keys = list(like.keys()) + ['_base']
            model, key = next((m, k) for m, k in zip(models, keys) if k in available_keys)
            document = MongoStyleDict(like)
            if '_outputs' not in document:
                document['_outputs'] = {}
            info = self.parent_if_appl['_objects'].find_one({'model': watcher_model,
                                                             'key': watcher_key,
                                                             'varieties': 'watcher'})
            features = info.get('features', {})
            for subkey in features:
                if subkey not in document:
                    continue
                if subkey not in document['_outputs']:
                    document['_outputs'][subkey] = {}
                if features[subkey] not in document['_outputs'][subkey]:
                    document['_outputs'][subkey][features[subkey]] = \
                        apply_model(self.models[features[subkey]], document[subkey])
                document[subkey] = document['_outputs'][subkey][features[subkey]]
            model_input = document[key if key != '_base' else document]
            model = self.models[model]
            with torch.no_grad():
                h = apply_model(model, model_input, True)
        return hash_set.find_nearest_from_hash(h, n=n)

    def _find_similar_then_matches(self, filter, like, *args, raw=False, features=None, n=10,
                                   **kwargs):
        similar = self._find_nearest(like, n=n)
        filter = {
            '$and': [
                filter,
                {'_id': {'$in': similar['_ids']}}
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
                scores=dict(zip(similar['_ids'], similar['scores'])),
                **kwargs,
            )

    def _find_matches_then_similar(self, filter, like, *args, raw=False, features=None, n=10,
                                   **kwargs):
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
            similar = self._find_nearest(like, ids=ids, n=n)
        else:  # pragma: no cover
            similar = self._find_nearest(like, n=n)
        if raw:
            return Cursor(self, {'_id': {'$in': similar['_ids']}}, **kwargs)  # pragma: no cover
        else:
            return SuperDuperCursor(self, {'_id': {'$in': similar['_ids']}},
                                    features=features,
                                    scores=dict(zip(similar['_ids'], similar['scores'])), **kwargs)

    def _gather_urls(self, documents):
        urls = []
        mongo_keys = []
        ids = []
        for r in documents:
            sub_urls, sub_mongo_keys = self._gather_urls_for_document(r)
            ids.extend([r['_id'] for _ in sub_urls])
            urls.extend(sub_urls)
            mongo_keys.extend(sub_mongo_keys)
        return urls, mongo_keys, ids

    @staticmethod
    def _gather_urls_for_document(r):
        '''
        >>> Collection._gather_urls_for_document({'a': {'_content': {'url': 'test'}}})
        (['test'], ['a'])
        >>> d = {'b': {'a': {'_content': {'url': 'test'}}}}
        >>> Collection._gather_urls_for_document(d)
        (['test'], ['b.a'])
        >>> d = {'b': {'a': {'_content': {'url': 'test', 'bytes': b'abc'}}}}
        >>> Collection._gather_urls_for_document(d)
        ([], [])
        '''
        urls = []
        keys = []
        for k in r:
            if isinstance(r[k], dict) and '_content' in r[k]:
                if 'url' in r[k]['_content'] and 'bytes' not in r[k]['_content']:
                    keys.append(k)
                    urls.append(r[k]['_content']['url'])
            elif isinstance(r[k], dict) and '_content' not in r[k]:
                sub_urls, sub_keys = Collection._gather_urls_for_document(r[k])
                urls.extend(sub_urls)
                keys.extend([f'{k}.{key}' for key in sub_keys])
        return urls, keys

    def _infer_types(self, documents):
        for r in documents:
            self._convert_types(r)
        return documents

    def _load_model(self, name):
        manifest = self.parent_if_appl[f'_objects'].find_one({'name': name, 'varieties': 'model'})
        if manifest is None:
            raise Exception(f'No such object of type "model", "{name}" has been registered.') # pragma: no cover
        model = self._load_object('model', name)
        if manifest['preprocessor'] is None and manifest['postprocessor'] is None:
            return model
        assert manifest.get('preprocessor') is not None \
            or manifest.get('forward') is not None
        preprocessor = None
        if manifest.get('preprocessor') is not None:
            preprocessor = self._load_object('preprocessor', manifest['preprocessor'])
        postprocessor = None
        if manifest.get('postprocessor') is not None:
            postprocessor = self._load_object('postprocessor', manifest['postprocessor'])
        return create_container(preprocessor, model, postprocessor)

    def _load_object(self, type, name):
        manifest = self.parent_if_appl[f'_objects'].find_one({'name': name, 'varieties': type})
        if manifest is None:
            raise Exception(f'No such object of type "{type}", "{name}" has been registered.') # pragma: no cover
        m = self.parent_if_appl._load_pickled_file(manifest['object'])
        if isinstance(m, torch.nn.Module):
            m.eval()
        return m

    def _load_hashes(self, name):
        si = self.parent_if_appl['_objects'].find_one({'name': name, 'varieties': 'semantic_index'})
        watcher_info = \
            self.parent_if_appl['_objects'].find_one({'model': si['models'][0],
                                                      'key': si['keys'][0],
                                                      'varieties': 'watcher'})
        filter = watcher_info.get('filter', {})
        key = watcher_info.get('key', '_base')
        filter[f'_outputs.{key}.{watcher_info["model"]}'] = {'$exists': 1}
        n_docs = self.count_documents(filter)
        c = self.find(filter, {f'_outputs.{key}.{watcher_info["model"]}': 1})
        measure = self.measures[si['measure']]
        loaded = []
        ids = []
        docs = progressbar(c, total=n_docs)
        print(f'loading hashes: "{name}"')
        for r in docs:
            h = MongoStyleDict(r)[f'_outputs.{key}.{watcher_info["model"]}']
            loaded.append(h)
            ids.append(r['_id'])
        return hashes.HashSet(torch.stack(loaded), ids, measure=measure)

    def _load_pickled_file(self, file_id):
        return loading.load(file_id, filesystem=self.filesystem)

    def _get_dependencies_for_watcher(self, model, key):
        info = self['_objects'].find_one({'model': model, 'key': key, 'varieties': 'watcher'},
                                         {'features': 1})
        if info is None:
            return []
        model_features = info.get('features', {})
        return list(zip(model_features.values(), model_features.keys()))

    def _get_watcher_for_neighbourhood(self, neigh):
        info = self['_objects'].find_one({'name': neigh, 'varieties': 'neighbourhood'})
        watcher_info = self['_objects'].find_one({'name': info['watcher'], 'varieties': 'watcher'})
        return (watcher_info['model'], watcher_info['key'])

    def _submit_process_documents_with_watcher(self, model, key, sub_ids, dependencies,
                                               verbose=True):
        watcher_info = \
            self.parent_if_appl['_objects'].find_one({'model': model, 'varieties': 'watcher',
                                                      'key': key})
        if not self.remote:
            self._process_documents_with_watcher(
                model_name=model, key=key, ids=sub_ids, verbose=verbose,
            )
            if watcher_info.get('download', False):  # pragma: no cover
                self._download_content(ids=sub_ids)
        else:
            return superduper_requests.jobs.process(
                self.parent_if_appl.database.name,
                self.name,
                '_process_documents_with_watcher',
                model_name=model,
                ids=sub_ids,
                verbose=verbose,
                dependencies=dependencies,
            )

    def _create_filter_lookup(self, ids):
        filters = []
        for model, key in self.list_watchers():
            watcher_info = self.parent_if_appl['_objects'].find_one({'model': model, 'key': key,
                                                                     'varieties': 'watcher'})
            filters.append(watcher_info.get('filter') or {})
        filter_lookup = {self._dict_to_str(f): f for f in filters}
        lookup = {}
        for filter_str in filter_lookup:
            if filter_str not in lookup:
                tmp_ids = [
                    r['_id']
                    for r in super().find({
                        '$and': [{'_id': {'$in': ids}}, filter_lookup[filter_str]]
                    })
                ]
                lookup[filter_str] = {'_ids': tmp_ids}
        return lookup

    def _submit_download_content(self, ids, dependencies=()):
        if not self.remote:
            print('downloading content from retrieved urls')
            self._download_content(ids=ids)
        else:
            return superduper_requests.jobs.process(
                self.database.name,
                self.name,
                '_download_content',
                ids=ids,
                dependencies=dependencies,
            )

    def _submit_compute_neighbourhood(self, item, sub_ids, dependencies):
        if not self.remote:
            self._compute_neighbourhood(item, sub_ids)
        else:
            return superduper_requests.jobs.process(
                self.database.name,
                self.name,
                '_compute_neighbourhood',
                name=item,
                ids=sub_ids,
                dependencies=dependencies
            )

    def _process_single_item(self, type_, item, iteration, lookup, job_ids, download_id,
                             verbose=True):
        if type_ == 'watcher':
            watcher_info = self.parent_if_appl['_objects'].find_one(
                {'model': item[0], 'key': item[1], 'varieties': 'watcher'}
            )
            if iteration == 0:
                dependencies = [download_id]
            else:
                model_dependencies = \
                    self.parent_if_appl._get_dependencies_for_watcher(item)
                dependencies = sum([
                    job_ids[('models', dep)]
                    for dep in model_dependencies
                ], [])
            filter_str = self._dict_to_str(watcher_info.get('filter') or {})
            sub_ids = lookup[filter_str]['_ids']
            process_id = \
                self._submit_process_documents_with_watcher(item[0], item[1], sub_ids, dependencies,
                                                            verbose=verbose)
            job_ids[(type_, item)].append(process_id)
            if watcher_info.get('download', False):  # pragma: no cover
                download_id = \
                    self._submit_download_content(sub_ids, dependencies=(process_id,))
                job_ids[(type_, item)].append(download_id)
        elif type_ == 'neighbourhoods':
            model = self.parent_if_appl._get_watcher_for_neighbourhood(item)
            watcher_info = self.parent_if_appl['_objects'].find_one({'name': model, 'varieties': 'watcher'})
            filter_str = self._dict_to_str(watcher_info.get('filter') or {})
            sub_ids = lookup[filter_str]['_ids']
            dependencies = job_ids[('models', model)]
            process_id = self._submit_compute_neighbourhood(item, sub_ids, dependencies)
            job_ids[(type_, item)].append(process_id)
        return job_ids

    def _process_documents(self, ids, verbose=False):
        job_ids = defaultdict(lambda: [])
        download_id = self._submit_download_content(ids=ids)
        if not self.list_watchers():
            return
        lookup = self._create_filter_lookup(ids)
        G = self._create_plan()
        current = [('watcher', watcher) for watcher in self.list_watchers()
                   if not list(G.predecessors(('watcher', watcher)))]
        iteration = 0
        while current:
            for (type_, item) in current:
                job_ids = self._process_single_item(type_, item, iteration, lookup, job_ids,
                                                    download_id, verbose=verbose)
            current = sum([list(G.successors((type_, item))) for (type_, item) in current], [])
            iteration += 1
        return job_ids

    def _compute_model_outputs(self, ids, model_info, features=None,
                               key='_base', model=None, verbose=True, loader_kwargs=None):
        print('finding documents under filter')
        features = features or {}
        model_name = model_info['name']
        if features is None:
            features = {}  # pragma: no cover

        documents = self.find({'_id': {'$in': ids}}, features=features)
        documents = list(documents)
        ids = [r['_id'] for r in documents]  # find statement messes with the ordering
        for r in documents:
            del r['_id'] # _id can't be handled by dataloader
        print('done.')
        if key != '_base' or '_base' in features:
            passed_docs = [r[key] for r in documents]
        else:  # pragma: no cover
            passed_docs = documents
        if model is None:  # model is not None during training, since a suboptimal model may be in need of validation
            model = self.models[model_name]
        inputs = training.loading.BasicDataset(
            passed_docs,
            model.preprocess if hasattr(model, 'preprocess') else lambda x: x
        )
        loader_kwargs = loader_kwargs or {}
        if isinstance(model, torch.nn.Module):
            loader = torch.utils.data.DataLoader(
                inputs,
                **loader_kwargs,
            )
            if verbose:
                print(f'processing with {model_name}')
                loader = progressbar(loader)
            outputs = []
            has_post = hasattr(model, 'postprocess')
            for batch in loader:
                with torch.no_grad():
                    output = model.forward(batch)
                if has_post:
                    unpacked = unpack_batch(output)
                    outputs.extend([model.postprocess(x) for x in unpacked])
                else:
                    outputs.extend(unpack_batch(output))
        else:
            outputs = []
            num_workers = loader_kwargs.get('num_workers', 0)
            if num_workers:
                pool = multiprocessing.Pool(processes=num_workers)
                for r in pool.map(model.preprocess, passed_docs):
                    outputs.append(r)
                pool.close()
                pool.join()
            else:
                for r in passed_docs:  # pragma: no cover
                    outputs.append(model.preprocess(r))
        return outputs, ids

    def _process_documents_with_watcher(self, model_name, key, ids, verbose=False,
                                        max_chunk_size=5000, model=None, recompute=False):
        import sys
        sys.path.insert(0, os.getcwd())

        watcher_info = self.parent_if_appl['_objects'].find_one(
            {'model': model_name, 'key': key, 'varieties': 'watcher'}
        )
        if not recompute:
            ids = [r['_id'] for r in self.find({'_id': {'$in': ids},
                                                f'_outputs.{key}.{model_name}': {'$exists': 0}})]
        if max_chunk_size is not None:
            for it, i in enumerate(range(0, len(ids), max_chunk_size)):
                print('computing chunk '
                      f'({it + 1}/{math.ceil(len(ids) / max_chunk_size)})')
                self._process_documents_with_watcher(
                    model_name,
                    key,
                    ids[i: i + max_chunk_size],
                    verbose=verbose,
                    max_chunk_size=None,
                    model=model,
                    recompute=recompute,
                )
            return

        model_info = self.parent_if_appl['_objects'].find_one({'name': model_name, 'varieties': 'model'})
        outputs, ids = self._compute_model_outputs(ids,
                                                   model_info,
                                                   key=key,
                                                   features=watcher_info.get('features', {}),
                                                   model=model,
                                                   loader_kwargs=watcher_info.get('loader_kwargs'),
                                                   verbose=verbose)

        type_ = self.parent_if_appl['_objects'].find_one({'name': model_name, 'varieties': 'model'},
                                                         {'type': 1}).get('type')
        if type_:
            type_ = self.types[type_]
            outputs = [
                {
                    '_content': {
                        'bytes': type_.encode(x),
                        'type': model_info['type']
                    }
                }
                for x in outputs
            ]

        self._write_watcher_outputs(outputs, ids, watcher_info)
        return outputs

    def _write_watcher_outputs(self, outputs, ids, watcher_info):
        key = watcher_info.get('key', '_base')
        model_name = watcher_info['model']
        print('bulk writing...')
        if watcher_info.get('target') is None:
            self.bulk_write([
                UpdateOne({'_id': id},
                          {'$set': {f'_outputs.{key}.{model_name}': outputs[i]}})
                for i, id in enumerate(ids)
            ])
        else:  # pragma: no cover
            self.bulk_write([
                UpdateOne({'_id': id},
                          {'$set': {
                              watcher_info['target']: outputs[i]
                          }})
                for i, id in enumerate(ids)
            ])
        print('done.')

    @staticmethod
    def _remove_like_from_filter(r):
        return {k: v for k, v in r.items() if k != '$like'}

    def _replace_model(self, name, object):
        r = self['_objects'].find_one({'name': name, 'varieties': 'model'})
        assert name in self.list_models(), f'model "{name}" doesn\'t exist to replace'
        if isinstance(r['object'], ObjectId):
            file_id = self._create_pickled_file(object)
            self.filesystem.delete(r['object'])
            self['_objects'].update_one({'name': name, 'varieties': 'model'},
                                        {'$set': {'object': file_id}})
        elif isinstance(r['object'], str):
            self._replace_model(r['object'], object)
        else:
            assert r['object'] is None
            if isinstance(r['preprocessor'], str):
                file_id = self._create_pickled_file(object._preprocess)
                pre_info = self['_objects'].find_one({'name': r['preprocessor'], 'varieties': 'preprocessor'})
                self.filesystem.delete(pre_info['object'])
                self['_objects'].update_one(
                    {'name': r['preprocessor'], 'varieties': 'function'},
                    {'$set': {'object': file_id}}
                )

            if isinstance(r['forward'], str):
                file_id = self._create_pickled_file(object._forward)
                forward_info = self['_objects'].find_one({'name': r['forward'], 'varieties': 'model'})
                self.filesystem.delete(forward_info['object'])
                self['_objects'].update_one({'name': r['forward'], 'varieties': 'model'}, {'$set': {'object': file_id}})

            if isinstance(r['postprocessor'], str):
                file_id = self._create_pickled_file(object._postprocess)
                post_info = self['_objects'].find_one({'name': r['postprocessor'], 'varieties': 'postprocessor'})
                self.filesystem.delete(post_info['object'])
                self['_objects'].update_one({'name': r['postprocessor'], 'varieties': 'postprocessor'},
                                            {'$set': {'object': file_id}})

    @staticmethod
    def _standardize_dict(d):  # pragma: no cover
        keys = sorted(list(d.keys()))
        out = {}
        for k in keys:
            if isinstance(d[k], dict):
                out[k] = Collection._standardize_dict(d[k])
            else:
                out[k] = d[k]
        return out

    def _train_imputation(self, name):

        import sys
        sys.path.insert(0, os.getcwd())

        if self.remote:
            return superduper_requests.jobs.process(self.database.name, self.name,
                                                    '_train_imputation', name)

        info = self['_objects'].find_one({'name': name, 'varieties': 'imputation'})
        splitter = None
        if info.get('splitter'):
            splitter = self.splitters[info['splitter']]

        model = self.models[info['model']]
        target = self.functions[info['target']]
        objective = self.objectives[info['objective']]
        metrics = {k: self.metrics[k] for k in info['metrics']}
        keys = (info['model_key'], info['target_key'])

        training.train_tools.ImputationTrainer(
            name,
            cf['mongodb'],
            self.database.name,
            self.name,
            models=(model, target),
            keys=keys,
            model_names=(info['model'], info['target']),
            objective=objective,
            metrics=metrics,
            **info['trainer_kwargs'],
            save=self._replace_model,
            splitter=splitter,
        ).train()

    def _train_semantic_index(self, name):

        import sys
        sys.path.insert(0, os.getcwd())

        if self.remote:
            return superduper_requests.jobs.process(self.database.name, self.name,
                                                    '_train_semantic_index', name)

        info = self['_objects'].find_one({'name': name, 'varieties': 'semantic_index'})
        model_names = info['models']

        models = []
        for mn in info['models']:
            models.append(self.models[mn])

        metrics = {}
        for metric in info['metrics']:
            metrics[metric] = self.metrics[metric]

        objective = self.objectives[info['objective']]
        splitter = info.get('splitter')
        if splitter:
            splitter = self.splitters[info['splitter']]

        t = training.train_tools.SemanticIndexTrainer(
            name,
            cf['mongodb'],
            self.database.name,
            self.name,
            models=models,
            keys=info['keys'],
            model_names=info['models'],
            splitter=splitter,
            objective=objective,
            save=self._replace_model,
            watch='objective',
            metrics=metrics,
            validation_sets=info.get('validation_sets', ()),
            **info.get('trainer_kwargs', {}),
        )
        t.train()
