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
from pymongo.database import Database as MongoDatabase
import superduperdb.mongodb.collection
from superduperdb import training, cf
from superduperdb.getters import jobs
from superduperdb.getters import client as our_client
from superduperdb.models import loading
from superduperdb.models.utils import create_container, apply_model, BasicDataset
from superduperdb.training.validation import validate_representations
from superduperdb.types.utils import convert_types
from superduperdb.utils import ArgumentDefaultDict, progressbar, unpack_batch, Downloader, \
    MongoStyleDict


class Database(MongoDatabase):
    """
    Database building on top of :code:`pymongo.database.Database`. Collections in the
    database are SuperDuperDB objects :code:`superduperdb.collection.Collection`.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._filesystem = None
        self._filesystem_name = f'_{self.name}:files'
        self._type_lookup = None

        self.models = ArgumentDefaultDict(lambda x: self._load_model(x))
        self.functions = ArgumentDefaultDict(lambda x: self._load_object('function', x))
        self.preprocessors = ArgumentDefaultDict(lambda x: self._load_object('preprocessor', x))
        self.postprocessors = ArgumentDefaultDict(lambda x: self._load_object('postprocessor', x))
        self.types = ArgumentDefaultDict(lambda x: self._load_object('type', x))
        self.splitters = ArgumentDefaultDict(lambda x: self._load_object('splitter', x))
        self.objectives = ArgumentDefaultDict(lambda x: self._load_object('objective', x))
        self.measures = ArgumentDefaultDict(lambda x: self._load_object('measure', x))
        self.metrics = ArgumentDefaultDict(lambda x: self._load_object('metric', x))

        self.remote = cf.get('remote', False)

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

    def __getitem__(self, name: str):
        if name != '_validation_sets' and name.startswith('_'):
            return super().__getitem__(name)
        return superduperdb.mongodb.collection.Collection(self, name)

    @property
    def filesystem(self):
        if self._filesystem is None:
            self._filesystem = gridfs.GridFS(
                self.client[self._filesystem_name]
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
        assert name not in self['_objects'].distinct('name', {'variety': 'model'}), \
            f'Model {name} already exists!'

        if type is not None:
            assert type in self['_objects'].distinct('name', {'variety': 'type'})

        if isinstance(object, str):
            file_id = object
        else:
            file_id = self._create_pickled_file(object)

        self['_objects'].insert_one({
            'variety': 'model',
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
            'variety': 'neighbourhood',
        })
        info = self['_objects'].find_one({'name': watcher})
        watcher_info = list(self['_objects'].find({'model': {'$in': info['models']},
                                                   'variety': 'watcher'}))[0]
        filter_ = watcher_info.get('filter', {})
        ids = [r['_id'] for r in self.find(filter_, {'_id': 1})]
        if not self.remote:
            self._compute_neighbourhood(collection, name, ids)
        else:
            return superduper_requests.jobs.process(
                self.name,
                '_compute_neighbourhood',
                collection,
                name,
                ids=ids,
            )

    def _download_content(self, collection, ids=None, documents=None, timeout=None, raises=True,
                          n_download_workers=None, headers=None):
        import sys
        sys.path.insert(0, os.getcwd())

        collection = self[collection]

        update_db = False
        if documents is None:
            update_db = True
            assert ids is not None
            documents = list(collection.find({'_id': {'$in': ids}}, {'_outputs': 0}, raw=True))
        urls, keys, place_ids = collection._gather_urls(documents)
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
            collection=collection,
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

    def _submit_download_content(self, collection, ids, dependencies=()):
        if not self.remote:
            print('downloading content from retrieved urls')
            self._download_content(collection, ids=ids)
        else:
            return jobs.process(
                self.name,
                '_download_content',
                collection=collection,
                ids=ids,
                dependencies=dependencies,
            )

    def create_watcher(self, collection, model, key='_base', filter_=None, verbose=False, target=None,
                       process_docs=True, features=None, loader_kwargs=None,
                       dependencies=(), superduper_requests=None):

        assert self['_objects'].count_documents({'model': model,
                                                 'key': key,
                                                 'collection': collection,
                                                 'variety': 'watcher'}) == 0, \
            f"This watcher {model}, {key} already exists"

        self['_objects'].insert_one({
            'variety': 'watcher',
            'model': model,
            'filter': filter_ if filter_ else {},
            'collection': collection,
            'key': key,
            'features': features if features else {},
            'target': target,
            'loader_kwargs': loader_kwargs or {},
        })

        if process_docs:
            ids = [r['_id'] for r in self[collection].find(filter_ if filter_ else {}, {'_id': 1})]
            if not ids:
                return
            if not self.remote:
                self._process_documents_with_watcher(collection, model, key, ids, verbose=verbose)
            else:  # pragma: no cover
                return jobs.process(
                    self.name,
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

        assert name not in self.list_imputations(collection)
        assert target in self.list_functions() or target in self.list_models()
        assert model in self.list_models()
        if objective is not None:
            assert objective in self.list_objectives()
        if metrics:
            for metric in metrics:
                assert metric in self.list_metrics()

        self['_objects'].insert_one({
            'variety': 'imputation',
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
            jobs = [self._train_imputation(collection, name)]
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
        assert name not in self.list_semantic_indexes(collection)

        if objective is not None:  # pragma: no cover
            if len(models) == 1:
                assert splitter is not None, 'need a splitter for self-supervised ranking...'

        self['_objects'].insert_one({
            'variety': 'semantic_index',
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
        self.create_watcher(collection, models[0], keys[0], filter_=filter_, process_docs=False,
                            features=trainer_kwargs.get('features', {}),
                            loader_kwargs=loader_kwargs or {})
        if objective is None:
            return [self.refresh_watcher(collection, models[0], keys[0], dependencies=())]
        try:
            jobs = [self._train_semantic_index(collection, name=name)]
        except KeyboardInterrupt:
            print('training aborted...')
            jobs = []
        jobs.append(self.refresh_watcher(collection, models[0], keys[0], dependencies=jobs))
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
            total = self[collection].count_documents(filter)
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
            r['_collection'] = collection
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

        info = self['_objects'].find_one({'name': name, 'variety': 'imputation',
                                          'collection': collection})
        if info is None and force:
            return
        self['_objects'].delete_one({'model': info['model'], 'key': info['model_key'],
                                     'variety': 'watcher', 'collection': collection})
        self['_objects'].delete_one({'name': name, 'variety': 'imputation',
                                     'collection': collection})

    def delete_watcher(self, collection, model, key, force=False, delete_outputs=True):
        """
        Delete model from collection

        :param collection: Collection name
        :param name: Name of model
        :param force: Toggle to ``True`` to skip confirmation
        """
        info = self['_objects'].find_one({'model': model, 'key': key, 'collection': collection,
                                          'variety': 'watcher'})
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
            self[collection].update_many(
                info.get('filter') or {},
                {'$unset': {f'_outputs.{info["key"]}.{info["model"]}': 1}},
                refresh=False,
            )
        return self['_objects'].delete_one({'model': model, 'key': key, 'collection': collection,
                                            'variety': 'watcher'})

    def delete_neighbourhood(self, collection, name, force=False):
        """
        Delete neighbourhood from collection documents.

        :param collection: Collection name.
        :param name: Name of neighbourhood.
        :param force: Toggle to ``True`` to skip confirmation.
        """
        info = self['_objects'].find_one({'name': name, 'variety': 'neighbourhood',
                                          'collection': collection})
        watcher_info = self['_objects'].find_one({'name': info['watcher'],
                                                  'variety': 'watcher',
                                                  'collection': collection})
        filter_ = watcher_info['filter']
        n_documents = self.count_documents(filter_)
        if force or click.confirm(f'Removing neighbourhood "{name}", this will affect {n_documents}'
                                  ' documents. Are you sure?', default=False):
            self['_objects'].delete_one({'name': name, 'variety': 'neighbourhood',
                                         'collection': collection})
            self.update_many(filter_, {'$unset': {f'_like.{name}': 1}}, refresh=False)
        else:
            print('aborting') # pragma: no cover

    def delete_semantic_index(self, collection, name, force=False):
        """
        Delete semantic index.

        :param collection: Name of collection
        :param name: Name of semantic index
        :param force: Toggle to ``True`` to skip confirmation
        """
        info = self['_objects'].find_one({'name': name, 'variety': 'semantic_index',
                                          'collection': collection})
        watcher = self['_objects'].find_one({'model': info['models'][0], 'key': info['keys'][0],
                                             'variety': 'watcher', 'collection': collection})
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
        self['_objects'].delete_one({'name': name, 'variety': 'semantic_index',
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

    def create_function(self, name, object):
        return self._create_object(name, object, 'function')

    def create_measure(self, name, object):
        return self._create_object(name, object, 'measure')

    def create_metric(self, name, object):
        return self._create_object(name, object, 'metric')

    def create_objective(self, name, object):
        return self._create_object(name, object, 'objective')

    def create_splitter(self, name, object):
        return self._create_object(name, object, 'splitter')

    def create_postprocessor(self, name, object):
        return self._create_object(name, object, 'postprocessor')

    def create_preprocessor(self, name, object):
        return self._create_object(name, object, 'preprocessor')

    def create_type(self, name, object):
        return self._create_object(name, object, 'type')

    def _create_object(self, name, object, variety, collection=None):
        collection = {'collection': collection} if collection else {}
        assert self['_objects'].count_documents({'name': name, 'variety': variety, **collection}) == 0, \
            f'An object with the name: "{name}" and variety "{variety}" already exists...'
        file_id = self._create_pickled_file(object)
        self[f'_objects'].insert_one({'name': name, 'object': file_id, 'variety': variety,
                                      **collection})

    def _create_pickled_file(self, object):
        return loading.save(object, filesystem=self.filesystem)

    def delete_function(self, name, force=False):
        return self._delete_object('function', name, force=force)

    def delete_model(self, name, force=False):
        return self._delete_object('model', name, force=force)

    def delete_objective(self, name, force=False):
        return self.delete_objective('objective', name, force=force)

    def _delete_object(self, name, variety, force=False, collection=None):
        collection = {'collection': collection} if collection else {}
        r = self['_objects'].find_one({'name': name, 'variety': variety, **collection})
        if not r:
            if not force:
                raise Exception(f'{variety} "{r}" does not exist...')
            return
        assert self['_objects'].find_one(
            {'name': name, 'variety': variety, **collection}
        )['variety'] == variety, \
            'can\'t delete object in this way, since used may multiple types of objects'

        if force or click.confirm(f'You are about to delete {variety}: {object}, are you sure?',
                                  default=False):
            r = self[f'_objects'].find_one({'name': object, 'variety': variety, **collection})
            self.filesystem.delete(r['object'])
            self[f'_objects'].delete_one({'name': r['name'], 'variety': variety, **collection})

    def delete_measure(self, name, force=False):
        return self._delete_object(name, ['measure'], force=force)

    def delete_metric(self, name, force=False):
        return self._delete_object(name, ['metric'], force=force)

    def delete_splitter(self, name, force=False):
        return self._delete_object(name, ['splitter'], force=force)

    def delete_postprocessor(self, name, force=False):
        return self._delete_object(name, ['postprocessor'], force=force)

    def delete_preprocessor(self, name, force=False):
        return self._delete_object(name, ['preprocessor'], force=force)

    def delete_type(self, name, force=False):
        return self._delete_object(name, ['type'], force=force)

    def list_functions(self):
        return self.list_objects('function')

    def list_imputations(self, collection):
        return self.list_objects('function', collection=collection)

    def list_jobs(self):
        return list(self['_jobs'].find())

    def list_measures(self):
        return self.list_objects('measure')

    def list_metrics(self):
        return self.list_objects('metric')

    def list_models(self):
        return self.list_objects('model')

    def list_neighbourhoods(self):
        return self.list_objects('neighbourhood')

    def list_objectives(self):
        return self.list_objects('objective')

    def list_splitters(self):
        return self.list_objects('splitter')

    def list_postprocessors(self):
        return self.list_objects('postprocessor')

    def list_preprocessors(self):
        return self.list_objects('preprocessor')

    def list_types(self):
        return self.list_objects('type')

    @staticmethod
    def _dict_to_str(d):
        sd = Database._standardize_dict(d)
        return str(sd)

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
        r = self['_objects'].find_one({'name': name, 'variety': 'model'})
        assert name in self.list_models(), f'model "{name}" doesn\'t exist to replace'
        if isinstance(r['object'], ObjectId):
            file_id = self._create_pickled_file(object)
            self.filesystem.delete(r['object'])
            self['_objects'].update_one({'name': name, 'variety': 'model'},
                                        {'$set': {'object': file_id}})
        elif isinstance(r['object'], str):
            self._replace_model(r['object'], object)
        else:
            assert r['object'] is None
            if isinstance(r['preprocessor'], str):
                file_id = self._create_pickled_file(object._preprocess)
                pre_info = self['_objects'].find_one({'name': r['preprocessor'],
                                                      'variety': 'preprocessor'})
                self.filesystem.delete(pre_info['object'])
                self['_objects'].update_one(
                    {'name': r['preprocessor'], 'variety': 'function'},
                    {'$set': {'object': file_id}}
                )

            if isinstance(r['forward'], str):
                file_id = self._create_pickled_file(object._forward)
                forward_info = self['_objects'].find_one({'name': r['forward'],
                                                          'variety': 'model'})
                self.filesystem.delete(forward_info['object'])
                self['_objects'].update_one({'name': r['forward'], 'variety': 'model'},
                                            {'$set': {'object': file_id}})

            if isinstance(r['postprocessor'], str):
                file_id = self._create_pickled_file(object._postprocess)
                post_info = self['_objects'].find_one({'name': r['postprocessor'],
                                                       'variety': 'postprocessor'})
                self.filesystem.delete(post_info['object'])
                self['_objects'].update_one({'name': r['postprocessor'], 'variety': 'postprocessor'},
                                            {'$set': {'object': file_id}})

    def _write_watcher_outputs(self, collection, outputs, ids, watcher_info):
        key = watcher_info.get('key', '_base')
        model_name = watcher_info['model']
        print('bulk writing...')
        if watcher_info.get('target') is None:
            self[collection].bulk_write([
                UpdateOne({'_id': id},
                          {'$set': {f'_outputs.{key}.{model_name}': outputs[i]}})
                for i, id in enumerate(ids)
            ])
        else:  # pragma: no cover
            self[collection].bulk_write([
                UpdateOne({'_id': id},
                          {'$set': {
                              watcher_info['target']: outputs[i]
                          }})
                for i, id in enumerate(ids)
            ])
        print('done.')

    def _process_documents_with_watcher(self, collection, model_name, key, ids, verbose=False,
                                        max_chunk_size=5000, model=None, recompute=False):
        import sys
        sys.path.insert(0, os.getcwd())

        watcher_info = self['_objects'].find_one(
            {'model': model_name, 'key': key, 'variety': 'watcher'}
        )
        if not recompute:
            ids = [r['_id'] for r in self[collection].find({'_id': {'$in': ids},
                                                            f'_outputs.{key}.{model_name}': {'$exists': 0}})]
        if max_chunk_size is not None:
            for it, i in enumerate(range(0, len(ids), max_chunk_size)):
                print('computing chunk '
                      f'({it + 1}/{math.ceil(len(ids) / max_chunk_size)})')
                self._process_documents_with_watcher(
                    collection,
                    model_name,
                    key,
                    ids[i: i + max_chunk_size],
                    verbose=verbose,
                    max_chunk_size=None,
                    model=model,
                    recompute=recompute,
                )
            return

        model_info = self['_objects'].find_one({'name': model_name, 'variety': 'model'})
        outputs, ids = self._compute_model_outputs(collection,
                                                   ids,
                                                   model_info,
                                                   key=key,
                                                   features=watcher_info.get('features', {}),
                                                   model=model,
                                                   loader_kwargs=watcher_info.get('loader_kwargs'),
                                                   verbose=verbose)

        type_ = self['_objects'].find_one({'name': model_name, 'variety': 'model'},
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

        self._write_watcher_outputs(collection, outputs, ids, watcher_info)
        return outputs

    def _compute_model_outputs(self, collection, ids, model_info, features=None,
                               key='_base', model=None, verbose=True, loader_kwargs=None):
        print('finding documents under filter')
        features = features or {}
        model_name = model_info['name']
        if features is None:
            features = {}  # pragma: no cover

        collection = self[collection]
        documents = collection.find({'_id': {'$in': ids}}, features=features)
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
        inputs = BasicDataset(
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

    def _process_documents(self, collection, ids, verbose=False):
        job_ids = defaultdict(lambda: [])
        download_id = self._submit_download_content(collection, ids=ids)
        job_ids['download'].append(download_id)
        if not self.list_watchers(collection):
            return job_ids
        lookup = self._create_filter_lookup(collection, ids)
        G = self._create_plan(collection)
        current = [('watcher', watcher) for watcher in self.list_watchers(collection)
                   if not list(G.predecessors(('watcher', watcher)))]
        iteration = 0
        while current:
            for (type_, item) in current:
                job_ids.update(self._process_single_item(collection, type_, item, iteration,
                                                         lookup, job_ids,
                                                         download_id, verbose=verbose))
            current = sum([list(G.successors((type_, item))) for (type_, item) in current], [])
            iteration += 1
        return job_ids

    def _train_imputation(self, collection, name):

        import sys
        sys.path.insert(0, os.getcwd())

        if self.remote:
            return jobs.process(self.name,
                                                    '_train_imputation', collection, name)

        info = self['_objects'].find_one({'name': name, 'variety': 'imputation'})
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
            self.name,
            collection,
            models=(model, target),
            keys=keys,
            model_names=(info['model'], info['target']),
            objective=objective,
            metrics=metrics,
            **info['trainer_kwargs'],
            save=self._replace_model,
            splitter=splitter,
        ).train()

    def _train_semantic_index(self, collection, name):

        import sys
        sys.path.insert(0, os.getcwd())

        if self.remote:
            return jobs.process(self.name,
                                                    '_train_semantic_index',
                                                    collection,
                                                    name)

        info = self['_objects'].find_one({'name': name, 'variety': 'semantic_index'})
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
            self.name,
            collection,
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

    def _process_single_item(self, collection, type_, item, iteration, lookup, job_ids, download_id,
                             verbose=True):
        if type_ == 'watcher':
            watcher_info = self['_objects'].find_one(
                {'model': item[0], 'key': item[1], 'variety': 'watcher', 'collection': collection}
            )
            if iteration == 0:
                dependencies = [download_id]
            else:
                model_dependencies = \
                    self._get_dependencies_for_watcher(*item, collection)
                dependencies = sum([
                    job_ids[('models', dep)]
                    for dep in model_dependencies
                ], [])
            filter_str = self._dict_to_str(watcher_info.get('filter') or {})
            sub_ids = lookup[filter_str]['_ids']
            process_id = \
                self._submit_process_documents_with_watcher(collection,
                                                            item[0], item[1], sub_ids, dependencies,
                                                            verbose=verbose)
            job_ids[(type_, item)].append(process_id)
            if watcher_info.get('download', False):  # pragma: no cover
                download_id = \
                    self._submit_download_content(sub_ids, dependencies=(process_id,))
                job_ids[(type_, item)].append(download_id)
        elif type_ == 'neighbourhoods':
            model = self._get_watcher_for_neighbourhood(item, collection=collection)
            watcher_info = self['_objects'].find_one({'name': model, 'variety': 'watcher',
                                                      'collection': collection})
            filter_str = self._dict_to_str(watcher_info.get('filter') or {})
            sub_ids = lookup[filter_str]['_ids']
            dependencies = job_ids[('models', model)]
            process_id = self._submit_compute_neighbourhood(collection, item, sub_ids, dependencies)
            job_ids[(type_, item)].append(process_id)
        return job_ids

    def _submit_compute_neighbourhood(self, collection, item, sub_ids, dependencies):
        if not self.remote:
            self._compute_neighbourhood(collection, item, sub_ids)
        else:
            return jobs.process(
                self.name,
                collection,
                '_compute_neighbourhood',
                collection=collection,
                name=item,
                ids=sub_ids,
                dependencies=dependencies
            )

    def _create_filter_lookup(self, collection, ids):
        filters = []
        for model, key, collection in self.list_watchers(collection):
            watcher_info = self['_objects'].find_one({'model': model, 'key': key,
                                                      'collection': collection,
                                                      'variety': 'watcher'})
            filters.append(watcher_info.get('filter') or {})
        filter_lookup = {self._dict_to_str(f): f for f in filters}
        lookup = {}
        for filter_str in filter_lookup:
            if filter_str not in lookup:
                tmp_ids = [
                    r['_id']
                    for r in self[collection].find({
                        '$and': [{'_id': {'$in': ids}}, filter_lookup[filter_str]]
                    })
                ]
                lookup[filter_str] = {'_ids': tmp_ids}
        return lookup

    def _load_pickled_file(self, file_id):
        return loading.load(file_id, filesystem=self.filesystem)

    def _get_dependencies_for_watcher(self, model, key, collection):
        info = self['_objects'].find_one({'model': model, 'key': key, 'variety': 'watcher',
                                          'collection': collection},
                                         {'features': 1})
        if info is None:
            return []
        watcher_features = info.get('features', {})
        return list(zip(watcher_features.values(), watcher_features.keys()))

    def _get_watcher_for_neighbourhood(self, neigh, collection):
        info = self['_objects'].find_one({'name': neigh, 'variety': 'neighbourhood',
                                          'collection': collection})
        watcher_info = self['_objects'].find_one({'key': info['key'],
                                                  'model': info['model'],
                                                  'variety': 'watcher',
                                                  'collection': collection})
        return (watcher_info['model'], watcher_info['key'])

    def _submit_process_documents_with_watcher(self, collection, model, key, sub_ids, dependencies,
                                               verbose=True):
        watcher_info = \
            self['_objects'].find_one({'model': model, 'variety': 'watcher', 'key': key})
        if not self.remote:
            self._process_documents_with_watcher(
                collection=collection, model_name=model, key=key, ids=sub_ids, verbose=verbose,
            )
            if watcher_info.get('download', False):  # pragma: no cover
                self[collection]._download_content(ids=sub_ids)
        else:
            return jobs.process(
                self.name,
                '_process_documents_with_watcher',
                collection,
                model_name=model,
                key=key,
                ids=sub_ids,
                verbose=verbose,
                dependencies=dependencies,
            )

    def _load_object(self, type, name):
        manifest = self[f'_objects'].find_one({'name': name, 'variety': type})
        if manifest is None:
            raise Exception(f'No such object of type "{type}", "{name}" has been registered.')  # pragma: no cover
        m = self._load_pickled_file(manifest['object'])
        if isinstance(m, torch.nn.Module):
            m.eval()
        return m

    def _load_model(self, name):
        manifest = self[f'_objects'].find_one({'name': name, 'variety': 'model'})
        if manifest is None:
            raise Exception(f'No such object of type "model", "{name}" has been registered.')  # pragma: no cover
        manifest = dict(manifest)
        if isinstance(manifest['object'], str):
            manifest['object'] = self['_objects'].find_one({'name': manifest['object'],
                                                            'variety': 'model'})['object']
        model = self._load_pickled_file(manifest['object'])
        model.eval()
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

    def _create_plan(self, collection):
        G = networkx.DiGraph()
        for watcher in self.list_watchers(collection):
            G.add_node(('watcher', watcher))
        for model, key, collection in self.list_watchers(collection):
            deps = self._get_dependencies_for_watcher(model, key, collection)
            for dep in deps:
                G.add_edge(('watcher', dep), ('watcher', (model, key, collection)))
        for neigh in self.list_neighbourhoods():
            model, key = self._get_watcher_for_neighbourhood(neigh, collection)
            G.add_edge(('watcher', (model, key, collection)), ('neighbourhood', (neigh, collection)))
        assert networkx.is_directed_acyclic_graph(G)
        return G

    def list_objects(self, variety=None, query=None, collection=None):
        collection = {'collection': collection} if collection else {}

        if query is None:
            query = {}
        if variety is not None:
            return self['_objects'].distinct('name', {'variety': variety, **query, **collection})
        else:
            return list(self['_objects'].find(query, {'name': 1, 'variety': 1, '_id': 0}))

    def list_semantic_indexes(self, collection, query=None):
        query = query or {}
        return [r['name'] for r in self['_objects'].find({**query, 'variety': 'semantic_index',
                                                          'collection': collection},
                                                         {'name': 1})]

    def list_watchers(self, collection, query=None):
        if query is None:
            query = {}
        items = self['_objects'].find({**query, 'variety': 'watcher', 'collection': collection},
                                      {'model': 1, 'key': 1, '_id': 0})
        return [(r['model'], r['key'], collection) for r in items]

    def list_validation_sets(self, collection=None):
        """
        List validation sets
        :return: list of validation sets
        """
        collection = {'collection': collection} if collection else {}
        return self['_validation_sets'].distinct('_validation_set', **collection)

    def _compute_neighbourhood(self, collection, name, ids):

        import sys
        sys.path.insert(0, os.getcwd())

        info = self['_objects'].find_one({'name': name, 'variety': 'neighbourhood'})
        print('getting hash set')
        collection = self[collection]
        h = collection._all_hash_sets[info['semantic_index']]
        print(h.shape)
        print(f'computing neighbours based on neighbour "{name}" and '
              f'index "{info["semantic_index"]}"')

        for i in progressbar(range(0, len(ids), info['batch_size'])):
            sub = ids[i: i + info['batch_size']]
            results = h.find_nearest_from_ids(sub, n=info['n'])
            similar_ids = [res['_ids'] for res in results]
            collection.bulk_write([
                UpdateOne({'_id': id_}, {'$set': {f'_like.{name}': sids}})
                for id_, sids in zip(sub, similar_ids)
            ])

    def apply_model(self, model, input_, **kwargs):
        if self.remote:
            return our_client.apply_model(self.name, model, input_, **kwargs)
        if isinstance(model, str):
            model = self.models[model]
        with torch.no_grad():
            return apply_model(model, input_, **kwargs)

    def validate_semantic_index(self, name, validation_sets, metrics):
        results = {}
        features = self['_objects'].find_one({'name': name,
                                              'variety': 'semantic_index'}).get('features')
        for vs in validation_sets:
            results[vs] = validate_representations(self, vs, name, metrics, features=features)
        for vs in results:
            for m in results[vs]:
                self['_objects'].update_one(
                    {'name': name, 'variety': 'semantic_index'},
                    {'$set': {f'final_metrics.{vs}.{m}': results[vs][m]}}
                )

    def refresh_watcher(self, collection, model, key, dependencies=()):
        """
        Recompute model outputs.

        :param model: Name of model.
        """
        info = self['_objects'].find_one(
            {'model': model, 'key': key, 'variety': 'watcher', 'collection': collection}
        )
        ids = self[collection].distinct('_id', info.get('filter') or {})
        return self._submit_process_documents_with_watcher(collection, model, key, sub_ids=ids,
                                                           dependencies=dependencies)

    def convert_types(self, r):
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
