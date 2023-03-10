import math
import multiprocessing
import os
import time
from collections import defaultdict

import click
import gridfs
import networkx
import torch
from bson import ObjectId
from numpy.random import permutation
from pymongo import UpdateOne
from pymongo.database import Database as BaseDatabase
import superduperdb.collection
from superduperdb import getters as superduper_requests, training, cf
from superduperdb.models import loading
from superduperdb.models.utils import create_container
from superduperdb.types.utils import convert_types
from superduperdb.utils import ArgumentDefaultDict, progressbar, unpack_batch


class Database(BaseDatabase):
    """
    Database building on top of :code:`pymongo.database.Database`. Collections in the
    database are SuperDuperDB objects :code:`superduperdb.collection.Collection`.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._filesystem = None
        self._filesystem_name = f'_{self.name}:files'
        self._type_lookup = None
        self.objects = ArgumentDefaultDict(
            lambda x: ArgumentDefaultDict(lambda y: self._load_object(x[:-1], y))
        )
        self.models = ArgumentDefaultDict(lambda x: self._load_model(x))

        self._allowed_varieties = ['model', 'function', 'preprocessor', 'postprocessor',
                                   'splitter', 'objective', 'measure', 'type', 'metric',
                                   'imputation', 'neighbourhood']

    def __getitem__(self, name: str):
        return superduperdb.collection.Collection(self, name)

    @property
    def filesystem(self):
        if self._filesystem is None:
            self._filesystem = gridfs.GridFS(
                self.database[self._filesystem_name]
            )
        return self._filesystem

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
        :param postprocessor: separate postprocessing
        :param type: type for converting model outputs back and forth from bytes
        """
        assert name not in self['_objects'].distinct('name', {'varieties': 'model'}), \
            f'Model {name} already exists!'

        assert name not in self['_objects'].distinct('name', {'varieties': 'function'}), \
            f'Function {name} already exists!'

        if type is not None:
            assert type in self['_objects'].distinct('name', {'varieties': 'type'})

        if isinstance(object, str):
            file_id = object
        else:
            file_id = self._create_pickled_file(object)

        self['_objects'].insert_one({
            'varieties': ['model', 'function'],
            'name': name,
            'object': file_id,
            'type': type,
            'preprocessor': preprocessor,
            'postprocessor': postprocessor,
        })

    def create_neighbourhood(self, collection, name, n=10, watcher=None, batch_size=100):

        assert name not in self.list_neighbourhoods(collection=collection)
        self['_objects'].insert_one({
            'name': name,
            'watcher': watcher,
            'n': n,
            'collection': collection,
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
                '_compute_neighbourhood',
                collection,
                name,
                ids=ids,
            )

    def create_watcher(self, collection, model, key='_base', filter_=None, verbose=False, target=None,
                       process_docs=True, features=None, loader_kwargs=None,
                       dependencies=(), superduper_requests=None):

        assert self['_objects'].count_documents({'model': model, 'key': key, 'collection': collection,
                                                 'varieties': 'watcher'}) == 0, \
            f"This watcher {model}, {key} already exists"

        self['_objects'].insert_one({
            'varieties': ['watcher'],
            'model': model,
            'filter': filter_ if filter_ else {},
            'collection': collection,
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
                self._process_documents_with_watcher(collection, model, key, ids, verbose=verbose)
            else:  # pragma: no cover
                return superduper_requests.jobs.process(
                    self.database.name,
                    '_process_documents_with_watcher',
                    collection,
                    model,
                    key,
                    ids,
                    verbose=verbose,
                    dependencies=dependencies,
                )

    def create_imputation(self, collection, name, model, model_key, target, target_key,
                          objective=None, metrics=None, filter_=None,
                          splitter=None, watch=True, loader_kwargs=None, **trainer_kwargs):
        """
        Create an imputation setup. This is any learning task where we have an input to the model
        compared to the target.

        :param collection: Collection name
        :param name: Name of imputation
        :param model: Model settings or model name
        :param model_key: Key for model to injest
        :param target: Target settings (input to ``create_model``) or target name
        :param target_key: Key for model to predict
        :param objective: Loss settings or objective
        :param metrics: List of metric settings or metric names
        :param filter_: Filter for the watcher which may result
        :param splitter: Splitter name to use to prepare data points for insertion to model
        :param loader_kwargs: Keyword-arguments for the watcher
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
            'collection': collection,
            'name': name,
            'model': model,
            'model_key': model_key,
            'target': target,
            'target_key': target_key,
            'metrics': metrics or [],
            'objective': objective,
            'splitter': splitter,
            'loader_kwargs': loader_kwargs,
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
                collection=collection,
                model=model,
                key=model_key,
                filter_=filter_ or {},
                features=trainer_kwargs.get('features', {}),
                dependencies=jobs,
                verbose=True,
            ))
        return jobs

    def create_semantic_index(self, collection, name, models, measure, keys, validation_sets=(),
                              metrics=(), objective=None, filter_=None, splitter=None,
                              loader_kwargs=None, **trainer_kwargs):
        """
        :param collection: Name of collection
        :param name: Name of index
        :param models: List of existing models
        :param measure: Measure name
        :param keys: Keys in incoming data to listen to
        :param validation_sets: Name of immutable validation set to be used to evaluate performance
        :param metrics: List of existing metrics,
        :param objective: Loss name
        :param filter_: Filter on which to train
        :param splitter: Splitter name
        :param loader_kwargs: Keyword arguments to be passed to
        :param trainer_kwargs: Keyword arguments to be passed to ``training.train_tools.RepresentationTrainer``
        :return: List of job identifiers if ``self.remote``
        """
        assert name not in self.list_semantic_indexes()

        if objective is not None:  # pragma: no cover
            if len(models) == 1:
                assert splitter is not None, 'need a splitter for self-supervised ranking...'

        self['_objects'].insert_one({
            'varieties': ['semantic_index'],
            'collection': collection,
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

    def create_validation_set(self, collection, name, filter=None, chunk_size=1000, splitter=None,
                              sample_size=None):
        filter = filter or {}
        existing = self.list_validation_sets()
        if name in existing:
            raise Exception(f'validation set {name} already exists!')
        if isinstance(splitter, str):
            splitter = self.splitters[splitter]
        filter['_fold'] = 'valid'
        if sample_size is None:
            cursor = self[collection].find(filter, {'_id': 0}, raw=True)
            total = self.count_documents(filter)
        else:
            assert isinstance(sample_size, int)
            _ids = self.distinct('_id', filter)
            if len(_ids) > sample_size:
                _ids = [_ids[int(i)] for i in permutation(sample_size)]
            cursor = self[collection].find({'_id': {'$in': _ids}}, {'_id': 0}, raw=True)
            total = sample_size
        it = 0
        tmp = []
        for r in progressbar(cursor, total=total):
            if splitter is not None:
                r, other = splitter(r)
                r['_other'] = other
                r['_fold'] = 'valid'
            r['_validation_set'] = name
            r['collection'] = collection
            tmp.append(r)
            it += 1
            if it % chunk_size == 0:
                self['_validation_sets'].insert_many(tmp)
                tmp = []
        if tmp:
            self['_validation_sets'].insert_many(tmp)

    def delete_imputation(self, collection, name, force=False):
        """
        Delete imputation from collection

        :param collection: Collection of imputation
        :param name: Name of imputation
        :param force: Toggle to ``True`` to skip confirmation
        """
        do_delete = False
        if force or click.confirm(f'Are you sure you want to delete the imputation "{name}"?',
                                  default=False):
            do_delete = True
        if not do_delete:
            return

        info = self['_objects'].find_one({'name': name, 'varieties': 'imputation',
                                          'collection': collection})
        if info is None and force:
            return
        self['_objects'].delete_one({'model': info['model'], 'key': info['model_key'],
                                     'varieties': 'watcher', 'collection': collection})
        self['_objects'].delete_one({'name': name, 'varieties': 'imputation',
                                     'collection': collection})

    def delete_watcher(self, collection, model, key, force=False, delete_outputs=True):
        """
        Delete model from collection

        :param name: Name of model
        :param force: Toggle to ``True`` to skip confirmation
        """
        info = self['_objects'].find_one({'model': model, 'key': key, 'collection': collection,
                                          'varieties': 'watcher'})
        if not force: # pragma: no cover
            n_documents = self[collection].count_documents(info.get('filter') or {})
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

    def delete_neighbourhood(self, collection, name, force=False):
        """
        Delete neighbourhood from collection documents.

        :param name: Name of neighbourhood
        :param force: Toggle to ``True`` to skip confirmation
        """
        info = self['_objects'].find_one({'name': name, 'varieties': 'neighbourhood',
                                          'collection': collection})
        watcher_info = self['_objects'].find_one({'name': info['watcher'],
                                                  'varieties': 'watcher',
                                                  'collection': collection})
        filter_ = watcher_info['filter']
        n_documents = self.count_documents(filter_)
        if force or click.confirm(f'Removing neighbourhood "{name}", this will affect {n_documents}'
                                  ' documents. Are you sure?', default=False):
            self['_objects'].delete_one({'name': name, 'varieties': 'neighbourhood',
                                         'collection': collection})
            self.update_many(filter_, {'$unset': {f'_like.{name}': 1}}, refresh=False)
        else:
            print('aborting') # pragma: no cover

    def delete_semantic_index(self, collection, name, force=False):
        """
        Delete semantic index.

        :param name: Name of semantic index
        :param force: Toggle to ``True`` to skip confirmation
        """
        info = self['_objects'].find_one({'name': name, 'varieties': 'semantic_index',
                                          'collection': collection})
        watcher = self['_objects'].find_one({'model': info['models'][0], 'key': info['keys'][0],
                                             'varieties': 'watcher', 'collection': collection})
        if info is None:  # pragma: no cover
            return
        do_delete = False
        if force or \
                click.confirm(f'Are you sure you want to delete this semantic index: "{name}"; '):
            do_delete = True

        if not do_delete:
            return

        if watcher:
            self.delete_watcher(collection, watcher['model'], watcher['key'], force=force)
        self['_objects'].delete_one({'name': name, 'varieties': 'semantic_index',
                                     'collection': collection})

    def _get_content_for_filter(self, filter):
        if '_id' not in filter:
            filter['_id'] = 0
        urls = self._gather_urls([filter])[0]
        if urls:
            filter = self._download_content(documents=[filter],
                                            timeout=None, raises=True)[0]
            filter = convert_types(filter, converters=self.types)
        return filter

    def __getattr__(self, item):
        if item.startswith('create_'):
            variety = item.split('_')[-1]
            assert variety in self._allowed_varieties
            def create_local(name, object):
                return self.create_object(name, object, [variety])
            return create_local
        if item.startswith('delete_'):
            variety = item.split('_')[-1]
            assert variety in self._allowed_varieties
            def delete_local(name, force=False):
                r = self['_objects'].find_one({'name': name, 'varieties': variety})
                if not r:
                    if not force:
                        assert f'{variety} "{r}" does not exist...'
                    return
                assert self['_objects'].find_one({'name': name, 'varieties': variety})['varieties'] == [variety], \
                    'can\'t delete object in this way, since used may multiple types of objects'
                return self.delete_object([variety], name, force=force)
            return delete_local
        if item.startswith('list'):
            variety = item.split('_')[-1][:-1]
            assert variety in self._allowed_varieties
            def list_local(query=None):
                return self.list_objects(variety, query)
            return list_local
        if item[:-1] in self._allowed_varieties and item[-1] == 's':
            return self.objects[item]
        raise AttributeError(item)

    def create_object(self, name, object, varieties):
        for variety in varieties:
            assert variety in self._allowed_varieties, \
                f'type "{variety}" not admissible; allowed types are {self._allowed_varieties}'
            assert self['_objects'].count_documents({'name': name, 'varieties': variety}) == 0, \
                f'An object with the name: "{name}" and variety "{variety}" already exists...'
        file_id = self._create_pickled_file(object)
        self[f'_objects'].insert_one({'name': name, 'object': file_id, 'varieties': varieties})

    def _create_pickled_file(self, object):
        return loading.save(object, filesystem=self.filesystem)

    def convert_types(self, r):
        """
        Convert non MongoDB types to bytes using types already registered with collection.

        :param r: record in which to convert types
        :return modified record
        """
        if isinstance(r, dict):
            for k in r:
                r[k] = self.convert_types(r[k])
            return r
        try:
            t = self.type_lookup[type(r)]
        except KeyError:
            t = None
        if t is not None:
            return {'_content': {'bytes': self.types[t].encode(r), 'type': t}}
        return r

    def delete_model(self, name, force=False):
        return self.delete_object(['model', 'function'], name, force=force)

    def delete_object(self, varieties, object, force=False):
        data = list(self[f'_objects'].find({'name': object, 'varieties': varieties}))
        if not data:
            if not force:
                raise Exception(f'This object does not exist...{object}')
            return
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
    def _standardize_dict(d):  # pragma: no cover
        keys = sorted(list(d.keys()))
        out = {}
        for k in keys:
            if isinstance(d[k], dict):
                out[k] = Database._standardize_dict(d[k])
            else:
                out[k] = d[k]
        return out

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
                pre_info = self['_objects'].find_one({'name': r['preprocessor'],
                                                      'varieties': 'preprocessor'})
                self.filesystem.delete(pre_info['object'])
                self['_objects'].update_one(
                    {'name': r['preprocessor'], 'varieties': 'function'},
                    {'$set': {'object': file_id}}
                )

            if isinstance(r['forward'], str):
                file_id = self._create_pickled_file(object._forward)
                forward_info = self['_objects'].find_one({'name': r['forward'],
                                                          'varieties': 'model'})
                self.filesystem.delete(forward_info['object'])
                self['_objects'].update_one({'name': r['forward'], 'varieties': 'model'},
                                            {'$set': {'object': file_id}})

            if isinstance(r['postprocessor'], str):
                file_id = self._create_pickled_file(object._postprocess)
                post_info = self['_objects'].find_one({'name': r['postprocessor'],
                                                       'varieties': 'postprocessor'})
                self.filesystem.delete(post_info['object'])
                self['_objects'].update_one({'name': r['postprocessor'], 'varieties': 'postprocessor'},
                                            {'$set': {'object': file_id}})

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
            del r['_id']  # _id can't be handled by dataloader
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

    def _process_documents(self, ids, verbose=False):
        job_ids = defaultdict(lambda: [])
        download_id = self._submit_download_content(ids=ids)
        job_ids['download'].append(download_id)
        if not self.list_watchers():
            return job_ids
        lookup = self._create_filter_lookup(ids)
        G = self._create_plan()
        current = [('watcher', watcher) for watcher in self.list_watchers()
                   if not list(G.predecessors(('watcher', watcher)))]
        iteration = 0
        while current:
            for (type_, item) in current:
                job_ids.update(self._process_single_item(type_, item, iteration, lookup, job_ids,
                                                         download_id, verbose=verbose))
            current = sum([list(G.successors((type_, item))) for (type_, item) in current], [])
            iteration += 1
        return job_ids

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
                    self.parent_if_appl._get_dependencies_for_watcher(*item)
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
                key=key,
                ids=sub_ids,
                verbose=verbose,
                dependencies=dependencies,
            )

    def _load_object(self, type, name):
        manifest = self.parent_if_appl[f'_objects'].find_one({'name': name, 'varieties': type})
        if manifest is None:
            raise Exception(f'No such object of type "{type}", "{name}" has been registered.') # pragma: no cover
        m = self.parent_if_appl._load_pickled_file(manifest['object'])
        if isinstance(m, torch.nn.Module):
            m.eval()
        return m

    def _load_model(self, name):
        manifest = self.parent_if_appl[f'_objects'].find_one({'name': name, 'varieties': 'model'})
        if manifest is None:
            raise Exception(f'No such object of type "model", "{name}" has been registered.') # pragma: no cover
        manifest = dict(manifest)
        if isinstance(manifest['object'], str):
            manifest['object'] = self['_objects'].find_one({'name': manifest['object'],
                                                            'varieties': 'model'})['object']
        model = self.parent_if_appl._load_pickled_file(manifest['object'])
        if manifest['preprocessor'] is None and manifest['postprocessor'] is None:
            return model
        preprocessor = None
        if manifest.get('preprocessor') is not None:
            preprocessor = self._load_object('preprocessor', manifest['preprocessor'])
        postprocessor = None
        if manifest.get('postprocessor') is not None:
            postprocessor = self._load_object('postprocessor', manifest['postprocessor'])
        return create_container(preprocessor, model, postprocessor)

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
