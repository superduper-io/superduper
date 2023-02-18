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

    - ``create_imputation``
    - ``create_loss``
    - ``create_metric``
    - ``create_model``
    - ``create_neighbourhood``
    - ``create_semantic_index``
    - ``create_splitter``
    - ``create_type``

    Deleting objects:

    - ``delete_imputation``
    - ``delete_loss``
    - ``delete_metric``
    - ``delete_model``
    - ``delete_neighbourhood``
    - ``delete_semantic_index``
    - ``delete_splitter``
    - ``delete_type``

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

    - ``list_models``
    - ``list_semantic_indexes``
    - ``list_imputations``
    - ``list_types``
    - ``list_losses``
    - ``list_metrics``
    - ``list_types``

    Watching jobs

    - ``watch_job``

    Key properties:

    - ``hash_set`` (in memory vectors for neighbourhood search)
    - ``losses`` (dictionary of losses)
    - ``measures`` (dictionary of measures)
    - ``metrics`` (dictionary of metrics)
    - ``models`` (dictionary of models)
    - ``remote`` (whether the client sends requests in thread or to a server)
    - ``splitters`` (dictionary of splitters)
    - ``types`` (dictionary of types)

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

        self.forwards = ArgumentDefaultDict(lambda x: self._load_object('forwards', x))
        self.losses = ArgumentDefaultDict(lambda x: self._load_object('losses', x))
        self.measures = ArgumentDefaultDict(lambda x: self._load_object('measures', x))
        self.metrics = ArgumentDefaultDict(lambda x: self._load_object('metrics', x))
        self.models = ArgumentDefaultDict(lambda x: self._load_model(x))
        self.postprocessors = ArgumentDefaultDict(lambda x: self._load_object('postprocessors', x))
        self.preprocessors = ArgumentDefaultDict(lambda x: self._load_object('preprocessors', x))
        self.splitters = ArgumentDefaultDict(lambda x: self._load_object('splitters', x))
        self.types = ArgumentDefaultDict(lambda x: self._load_object('types', x))

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

    def apply_model(self, name, r, **kwargs):
        if self.remote:
            return superduper_requests.client.apply_model(name, r, **kwargs)
        model = self.models[name]
        info = self['_models'].find_one({'name': name})
        key = info['key']
        if isinstance(r, dict):
            if info['features']:
                r = MongoStyleDict(r)
                for k in info['features']:
                    r[k] = r[f'_outputs.{k}.{info["features"][k]}']
            if key == '_base':
                return apply_model(model, r, single=True, **kwargs)
            else:
                return apply_model(model, MongoStyleDict(r)[key], single=True, **kwargs)
        else:
            assert isinstance(r, list)
            if info['features']:
                for i, rr in enumerate(r):
                    r[i] = MongoStyleDict(rr)
                    for k in info['features']:
                        r[i][k] = r[i][f'_outputs.{k}.{info["features"][k]}']
            if key == '_base':
                return apply_model(model, r, single=False, **kwargs)
            else:
                return apply_model(model, [MongoStyleDict(rr)[key] for rr in r], single=False,
                                   **kwargs)

    def create_forward(self, name, object):
        """
        Create a forward directly from a Python object.

        :param name: name of forward
        :param object: Python object
        """
        return self._create_object('forwards', name, object)

    def create_imputation(self, name, model, loss, target, metrics=None, filter=None,
                          projection=None, splitter=None, **trainer_kwargs):
        """
        Create an imputation setup. This is any learning task where we have an input to the model
        compared to the target.

        :param name: Name of imputation
        :param model: Model settings or model name
        :param loss: Loss settings or loss
        :param target: Target settings (input to ``create_model``) or target name
        :param metrics: List of metric settings or metric names
        :param filter: Filter on which to train the imputation
        :param projection: Projection of data to apply during training (use for efficiency)
        :param splitter: Splitter name to use to prepare data points for insertion to model
        :param trainer_kwargs: Keyword-arguments to forward to ``train_tools.ImputationTrainer``
        :return: job_ids of jobs required to create the imputation
        """

        assert target in self.list_models()
        assert model in self.list_models()
        assert loss in self.list_losses()
        if metrics:
            for metric in metrics:
                assert metric in self.list_metrics()

        if filter is None:
            filter = self['_models'].find_one({'name': model}, {'filter': 1})['filter']

        self['_imputations'].insert_one({
            'name': name,
            'model': model,
            'target': target,
            'metrics': metrics or [],
            'loss': loss,
            'projection': projection,
            'filter': filter,
            'splitter': splitter,
            'trainer_kwargs': trainer_kwargs,
        })

        job_id = self._train_imputation(name)
        r = self['_models'].find_one({'name': model})
        ids = [r['_id'] for r in self.find(r.get('filter', {}), {'_id': 1})]

        if not self.remote:
            self._process_documents_with_model(model, ids, **r.get('loader_kwargs', {}),
                                               verbose=True)
        else:
            return [job_id, superduper_requests.jobs.process(
                self.database.name,
                self.name,
                '_process_documents_with_model',
                model_name=model,
                ids=ids,
                **r.get('loader_kwargs', {}),
                dependencies=(job_id,),
                verbose=False,
            )]

    def create_loss(self, name, object):
        """
        Create a loss function directly from a Python object.

        :param name: name of loss function
        :param object: Python object
        """
        return self._create_object('losses', name, object)

    def create_measure(self, name, object):
        """
        Create a document measure directly from a Python object.

        :param name: name of measure
        :param object: Python object
        """
        return self._create_object('measures', name, object)

    def create_metric(self, name, object):
        """
        Create a metric directly from a Python object.

        :param name: name of metric
        :param object: Python object
        """
        return self._create_object('metrics', name, object)

    def create_model(self, name, object=None, preprocessor=None, forward=None,
                     postprocessor=None, filter=None, type=None, active=True,
                     key='_base', verbose=False,
                     process_docs=True, loader_kwargs=None, max_chunk_size=5000, features=None):
        """
        Create a model registered in the collection directly from a python session.
        The added model will then watch incoming records and add outputs computed on those
        records into the ``"_outputs"`` fields of the records. The model is then stored inside MongoDB and can
        be accessed using the ``SuperDuperClient``.

        :param name: name of model
        :param object: if specified the model object (pickle-able) else None if model already exists
        :param preprocessor: separate preprocessing
        :param forward: separate forward pass
        :param postprocessor: separate postprocessing
        :param filter: filter specifying which documents model acts on
        :param type: type for converting model outputs back and forth from bytes
        :param active: toggle to ``False`` if model should not actively process incoming data
        :param key: key in records on which model acts (default whole record "_base")
        :param verbose: toggle to ``True`` if processing on data is verbose
        :param semantic_index: toggle to ``True``
        :param process_docs: toggle to ``False`` if documents not to be processed by models
        :param loader_kwargs: kwargs to be passed to dataloader for model in processing
        :param max_chunk_size: maximum chunk size of documents to be held in memory simultaneously
        :param features: dictionary of features to be substituted from model outputs to record
        :param neighbourboods: neighbourhoods whose computation is required for model inputs
        """
        if loader_kwargs is None:
            loader_kwargs = {}

        if active and object is not None and not isinstance(object, str):
            assert hasattr(object, 'preprocess') or hasattr(object, 'forward')

        assert name not in self['_models'].distinct('name'), \
            f'Model {name} already exists!'

        if type is not None:
            assert type in self['_types'].distinct('name')

        if preprocessor is not None or forward is not None or postprocessor is not None:
            assert object is None
            file_id = None
        else:
            if object is not None and not isinstance(object, str):
                file_id = self._create_pickled_file(object)
            elif isinstance(object, str):
                file_id = object

            preprocessor = None
            forward = None
            postprocessor = None

        self['_models'].insert_one({
            'name': name,
            'object': file_id,
            'filter': filter if filter else {},
            'type': type,
            'active': active,
            'key': key,
            'loader_kwargs': loader_kwargs if loader_kwargs else {},
            'max_chunk_size': max_chunk_size,
            'features': features if features else {},
            'training': False,
            'preprocessor': preprocessor,
            'forward': forward,
            'postprocessor': postprocessor,
        })

        if process_docs and active:
            ids = [r['_id'] for r in self.find(filter if filter else {}, {'_id': 1})]
            if ids:
                if not self.remote:
                    self._process_documents_with_model(name, ids, verbose=verbose,
                                                       max_chunk_size=max_chunk_size)
                else:  # pragma: no cover
                    return superduper_requests.jobs.process(
                        self.database.name,
                        self.name,
                        '_process_documents_with_model',
                        name,
                        ids=ids,
                        verbose=verbose,
                        max_chunk_size=max_chunk_size
                    )

    def create_neighbourhood(self, name, n=10, semantic_index=None, batch_size=100):
        assert name not in self.list_neighbourhoods()
        if semantic_index is None:
            semantic_index = self['_meta'].find_one({'key': 'semantic_index'})['value']
        self['_neighbourhoods'].insert_one({
            'name': name,
            'semantic_index': semantic_index,
            'n': n,
            'batch_size': batch_size,
        })
        info = self['_semantic_indexes'].find_one({'name': semantic_index})
        model_name = self.list_models(**{'active': True, 'name': {'$in': info['models']}})[0]
        model_info = self['_models'].find_one({'name': model_name})
        filter_ = model_info.get('filter', {})
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

    def create_preprocessor(self, name, object):
        """
        Create a preprocessor directly from a Python object.

        :param name: name of preprocessor
        :param object: Python object
        """
        return self._create_object('preprocessors', name, object)

    def create_postprocessor(self, name, object):
        """
        Create a postprocessor directly from a Python object.

        :param name: name of postprocessor
        :param object: Python object
        """
        return self._create_object('postprocessors', name, object)

    def create_semantic_index(self, name, models, measure, validation_sets=(), metrics=(), loss=None,
                              splitter=None, filter=None, **trainer_kwargs):
        """
        :param name: Name of index
        :param models: List of existing models
        :param measure: Measure name
        :param validation_sets: Name of immutable validation set to be used to evaluate performance
        :param metrics: List of existing metrics,
        :param loss: Loss name
        :param splitter: Splitter name
        :param filter: Filter on which to train
        :param trainer_kwargs: Keyword arguments to be passed to ``training.train_tools.RepresentationTrainer``
        :return: List of job identifiers if ``self.remote``
        """

        if loss is not None:  # pragma: no cover
            if len(models) == 1:
                assert splitter is not None, 'need a splitter for self-supervised ranking...'

        self['_semantic_indexes'].insert_one({
            'name': name,
            'models': models,
            'metrics': metrics,
            'loss': loss,
            'measure': measure,
            'splitter': splitter,
            'filter': filter or {},
            'validation_sets': list(validation_sets),
            'trainer_kwargs': trainer_kwargs,
        })
        if loss is None:
            if self.remote and validation_sets:
                return [superduper_requests.jobs.process(
                    self.database.name,
                    self.name,
                    'validate_semantic_index',
                    name,
                    validation_sets,
                    metrics,
                )]
            elif validation_sets:
                return self.validate_semantic_index(name, validation_sets, metrics)
            return

        try:
            active_model = self.list_models(**{'active': True, 'name': {'$in': models}})[0]
        except IndexError:
            print('no active models, returning...')
            return

        filter_ = self['_models'].find_one({'name': active_model}, {'filter': 1})['filter']

        job_ids = []
        job_ids.append(self._train_semantic_index(name=name))
        active_models = self.list_models(**{'active': True, 'name': {'$in': models}})
        for model in active_models:
            model_info = self['_models'].find_one({'name': model})
            ids = [x['_id'] for x in self.find(filter_, {'_id': 1})]
            if not self.remote:
                self._process_documents_with_model(
                    model,
                    ids,
                    verbose=True,
                    **model_info.get('loader_kwargs', {}),
                    max_chunk_size=model_info.get('max_chunk_size'),
                )
                self.validate_semantic_index(name, validation_sets, metrics)
            else:
                job_ids.append(superduper_requests.jobs.process(
                    self.database.name,
                    self.name,
                    '_process_documents_with_model',
                    model,
                    ids=ids,
                    verbose=True,
                    **model_info.get('loader_kwargs', {}),
                    max_chunk_size=model_info.get('max_chunk_size', 5000),
                ))
                job_ids.append(superduper_requests.jobs.process(
                    self.database.name,
                    self.name,
                    'validate_semantic_index',
                    name,
                    validation_sets,
                    metrics,
                ))
        if not self.remote:
            return
        return job_ids

    def create_splitter(self, name, object):
        """
        Create a document splitter directly from a Python object.

        :param name: name of splitter
        :param object: Python object
        """
        return self._create_object('splitters', name, object)

    def create_type(self, name, object):
        """
        Create a type directly from a Python object.

        :param name: name of type
        :param object: Python object
        """
        return self._create_object('types', name, object)

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
            r['_validation_set'] = name
            tmp.append(r)
            it += 1
            if it % chunk_size == 0:
                self['_validation_sets'].insert_many(tmp)
                tmp = []
        if tmp:
            self['_validation_sets'].insert_many(tmp)

    def delete_type(self, type, force=False):
        """
        Delete type from collection

        :param name: Name of type
        :param force: Toggle to ``True`` to skip confirmation
        """
        return self._delete_objects('types', objects=[type], force=force)

    def delete_forward(self, forward, force=False):
        """
        Delete forward from collection

        :param name: Name of forward
        :param force: Toggle to ``True`` to skip confirmation
        """
        return self._delete_objects('forwards', objects=[forward], force=force)

    def delete_imputation(self, name, force=False):
        """
        Delete imputation from collection

        :param name: Name of imputation
        :param force: Toggle to ``True`` to skip confirmation
        """
        do_delete = False
        if force or click.confirm(f'Are you sure you want to delete the imputation {name}?',
                                  default=False):
            do_delete = True
        if not do_delete:
            return

        self['_imputations'].delete_one({'name': name})

    def delete_loss(self, loss, force=False):
        """
        Delete loss from collection

        :param name: Name of loss
        :param force: Toggle to ``True`` to skip confirmation
        """
        return self._delete_objects('losses', objects=[loss], force=force)

    def delete_measure(self, name, force=False):
        """
        Delete measure from collection

        :param name: Name of measure
        :param force: Toggle to ``True`` to skip confirmation
        """
        return self._delete_objects('measures', objects=[name], force=force)

    def delete_metric(self, metric=None, force=False):
        """
        Delete metric from collection

        :param name: Name of metric
        :param force: Toggle to ``True`` to skip confirmation
        """
        return self._delete_objects('metrics', objects=[metric], force=force)

    def delete_model(self, name, force=False):
        """
        Delete model from collection

        :param name: Name of model
        :param force: Toggle to ``True`` to skip confirmation
        """
        info = self['_models'].find_one({'name': name}, {'filter': 1})
        if not force: # pragma: no cover
            n_documents = self.count_documents(info.get('filter') or {})
        do_delete = False
        if force or click.confirm(f'Are you sure you want to delete this model: {name}; '
                                  f'{n_documents} documents will be affected.',
                                  default=False):
            do_delete = True
        if not do_delete:
            return

        info = self['_models'].find_one({'name': name})
        self['_models'].delete_one({'name': name})

        if isinstance(info['object'], ObjectId):
            _ = self._delete_objects(
                'models', objects=[info['object']], force=True,
            )
        print(f'unsetting output field _outputs.{info["key"]}.{info["name"]}')
        super().update_many(
            info.get('filter') or {},
            {'$unset': {f'_outputs.{info["key"]}.{info["name"]}': 1}}
        )
        self['_validation_sets'].update_many(
            info.get('filter') or {},
            {'$unset': {f'_outputs.{info["key"]}.{info["name"]}': 1}},
            refresh=False,
        )

    def delete_neighbourhood(self, name, force=False):
        """
        Delete neighbourhood from collection documents.

        :param name: Name of neighbourhood
        :param force: Toggle to ``True`` to skip confirmation
        """
        info = self['_neighbourhoods'].find_one({'name': name})
        si_info = self['_semantic_indexes'].find_one({'name': info['semantic_index']})
        model = self.list_models(**{'active': True, 'name': {'$in': si_info['models']}})[0]
        filter_ = self['_models'].find_one({'name': model}, {'filter': 1})['filter']
        n_documents = self.count_documents(filter_)
        if force or click.confirm(f'Removing neighbourhood "{name}", this will affect {n_documents}'
                                  ' documents. Are you sure?', default=False):
            self['_neighbourhoods'].delete_one({'name': name})
            self.update_many(filter_, {'$unset': {f'_like.{name}': 1}}, refresh=False)
        else:
            print('aborting') # pragma: no cover

    def delete_postprocessor(self, postprocessor, force=False):
        """
        Delete postprocess from collection

        :param name: Name of postprocess
        :param force: Toggle to ``True`` to skip confirmation
        """
        return self._delete_objects('postprocessors', objects=[postprocessor], force=force)

    def delete_preprocessor(self, preprocessor, force=False):
        """
        Delete preprocess from collection

        :param name: Name of preprocess
        :param force: Toggle to ``True`` to skip confirmation
        """
        return self._delete_objects('preprocessors', objects=[preprocessor], force=force)

    def delete_semantic_index(self, name, force=False):
        """
        Delete semantic index.

        :param name: Name of semantic index
        :param force: Toggle to ``True`` to skip confirmation
        """
        info = self['_semantic_indexes'].find_one({'name': name})
        if info is None:  # pragma: no cover
            return
        do_delete = False
        if force or click.confirm(f'Are you sure you want to delete this semantic index: {name}; '):
            do_delete = True

        if not do_delete:
            return

        self['_semantic_indexes'].delete_one({'name': name})

    def delete_splitter(self, splitter=None, force=False):
        return self._delete_objects('splitters', objects=[splitter], force=force)

    def _get_content_for_filter(self, filter):
        if '_id' not in filter:
            filter['_id'] = 0
        urls = self._gather_urls([filter])[0]
        if urls:
            filter = self._download_content(documents=[filter],
                                            timeout=None, raises=True)[0]
            filter = convert_types(filter, converters=self.types)
        return filter

    def find(self, filter=None, *args, similar_first=False, raw=False,
             features=None, download=False, similar_join=None, **kwargs):
        """
        Behaves like MongoDB ``find`` with exception of ``$like`` operator.

        :param filter: filter dictionary
        :param args: args passed to super()
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
        if download:
            filter = self._get_content_for_filter(filter)    # pragma: no cover
            if '_id' in filter:
                del filter['_id']
        like_place = self._find_like_operator(filter)
        assert (like_place is None or like_place == '_base')
        if like_place is not None:
            filter = MongoStyleDict(filter)
            if similar_first:
                return self._find_similar_then_matches(filter, *args, raw=raw,
                                                       features=features, **kwargs)
            else:
                return self._find_matches_then_similar(filter, *args, raw=raw,
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

    def list_types(self):
        """
        List types
        :return: list of types
        """
        return self.parent_if_appl['_types'].distinct('name')

    def list_forwards(self):
        """
        List forwards
        :return: list of forwards
        """
        return self.parent_if_appl['_forwards'].distinct('name')

    def list_imputations(self):
        """
        List imputations
        :return: list of imputations
        """
        return self.parent_if_appl['_imputations'].distinct('name')

    def list_jobs(self):
        """
        List jobs.
        """
        return self.parent_if_appl['_jobs'].distinct('identifier')

    def list_losses(self):
        """
        List losses
        :return: list of losses
        """
        return self.parent_if_appl['_losses'].distinct('name')

    def list_measures(self):
        """
        List measures
        :return: list of measures
        """
        return self.parent_if_appl['_measures'].distinct('name')

    def list_metrics(self):
        """
        List metrics
        :return: list of metrics
        """
        return self.parent_if_appl['_metrics'].distinct('name')

    def list_models(self, **kwargs):
        """
        List models
        :return: list of models
        """
        return self.parent_if_appl['_models'].distinct('name', kwargs)

    def list_neighbourhoods(self):
        """
        List neighbourhoods.
        """
        return self.parent_if_appl['_neighbourhoods'].distinct('name')

    def list_postprocessors(self):
        """
        List postprocessors
        :return: list of postprocessors
        """
        return self.parent_if_appl['_postprocessors'].distinct('name')

    def list_preprocessors(self):
        """
        List preprocessors
        :return: list of preprocessors
        """
        return self.parent_if_appl['_preprocessors'].distinct('name')

    def list_semantic_indexes(self):
        """
        List semantic_indexes
        :return: list of semantic_indexes
        """
        return self.parent_if_appl['_semantic_indexes'].distinct('name')

    def list_splitters(self):
        """
        List splitters
        :return: list of splitters
        """
        return self.parent_if_appl['_splitters'].distinct('name')

    def list_validation_sets(self):
        """
        List validation sets
        :return: list of validation sets
        """
        return self.parent_if_appl['_validation_sets'].distinct('_validation_set')

    def refresh_model(self, model_name):
        info = self.parent_if_appl['_models'].find_one({'name': model_name})
        ids = self.distinct('_id', info.get('filter') or {})
        self._process_documents_with_model(model_name, ids=ids, **(info.get('loader_kwargs') or {}))

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
                r = self['_jobs'].find_one({'identifier': identifier},
                                           {'stdout': 1, 'status': 1})
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
        features = self['_semantic_indexes'].find_one({'name': name}).get('features')
        for vs in validation_sets:
            results[vs] = validate_representations(self, vs, name, metrics, features=features)
        for vs in results:
            for m in results[vs]:
                self['_semantic_indexes'].update_one(
                    {'name': name},
                    {'$set': {f'final_metrics.{vs}.{m}': results[vs][m]}}
                )

    def _compute_neighbourhood(self, name, ids):

        import sys
        sys.path.insert(0, os.getcwd())

        info = self['_neighbourhoods'].find_one({'name': name})
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
        for model in self.list_models(active=True):
            G.add_node(('models', model))
        for model in self.list_models():
            deps = self._get_dependencies_for_model(model)
            for dep in deps:
                G.add_edge(('models', dep), ('models', model))
        for neigh in self.list_neighbourhoods():
            model = self._get_model_for_neighbourhood(neigh)
            G.add_edge(('models', model), ('neighbourhoods', neigh))
        assert networkx.is_directed_acyclic_graph(G)
        return G

    def _create_object(self, type, name, object):
        assert name not in self[f'_{type}'].distinct('name')
        file_id = self._create_pickled_file(object)
        self[f'_{type}'].insert_one({'name': name, 'object': file_id})

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

    def _delete_objects(self, type, objects, force=False):
        data = list(self[f'_{type}'].find({'name': {'$in': objects}}))
        data += list(self[f'_{type}'].find({'$or': [
            {'name': {'$regex': f'^{x}\.'}}
            for x in objects
        ]}))
        for k in objects:
            if k in getattr(self, type):
                del getattr(self, type)[k]

        if force or click.confirm(f'You are about to delete these {type}: {objects}, are you sure?',
                                  default=False):
            for r in self[f'_{type}'].find({'name': {'$in': objects}}):
                if '.' not in r['name']:
                    self.filesystem.delete(r['object'])
                self[f'_{type}'].delete_one({'name': r['name']})
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

    def _find_nearest(self, filter, ids=None):
        if self.remote:
            filter, _ = self._convert_types(filter)
            return superduper_requests.client._find_nearest(self.database.name, self.name, filter,
                                                            ids=ids)
        assert '$like' in filter
        if ids is None:
            hash_set = self.hash_set
        else:  # pragma: no cover
            hash_set = self.hash_set[ids]

        if '_id' in filter['$like']['document']:
            return hash_set.find_nearest_from_id(filter['$like']['document']['_id'],
                                                 n=filter['$like']['n'])
        else:
            models = self.parent_if_appl['_semantic_indexes'].find_one({'name': self.semantic_index})['models']
            available_keys = list(filter['$like']['document'].keys()) + ['_base']
            man = self.parent_if_appl['_models'].find_one({
                'key': {'$in': available_keys},
                'name': {'$in': models},
            })
            model = self.models[man['name']]
            document = MongoStyleDict(filter['$like']['document'])
            if '_outputs' not in document:
                document['_outputs'] = {}
            info = self.parent_if_appl['_models'].find_one({'name': man['name']})
            features = info.get('features', {})
            for key in features:
                if key not in document['_outputs']:
                    document['_outputs'][key] = {}
                if features[key] not in document['_outputs'][key]:
                    document['_outputs'][key][features[key]] = \
                        apply_model(self.models[features[key]], document[key])
                document[key] = document['_outputs'][key][features[key]]
            model_input = document[man['key']] if man['key'] != '_base' else document
            with torch.no_grad():
                h = apply_model(model, model_input, True)
        return hash_set.find_nearest_from_hash(h, n=filter['$like']['n'])

    @staticmethod
    def _find_like_operator(r):  # pragma: no cover
        """

        >>> Collection._find_like_operator({'$like': 1})
        '_base'
        >>> Collection._find_like_operator({'a': {'$like': 1}})
        'a'
        >>> Collection._find_like_operator({'a': {'b': {'$like': 1}}})
        'a.b'

        """
        if '$like' in r:
            return '_base'
        else:
            for k in r:
                if isinstance(r[k], dict):
                    like_place = Collection._find_like_operator(r[k])
                    if like_place is not None:
                        if like_place == '_base':
                            return k
                        else:
                            return f'{k}.{like_place}'

    def _find_similar_then_matches(self, filter, *args, raw=False, features=None, **kwargs):
        similar = self._find_nearest(filter)
        new_filter = self._remove_like_from_filter(filter)
        filter = {
            '$and': [
                new_filter,
                {'_id': {'$in': similar['_ids']}}
            ]
        }
        if raw:
            return Cursor(self, filter, *args, **kwargs) # pragma: no cover
        else:
            return SuperDuperCursor(
                self,
                filter,
                *args,
                features=features,
                scores=dict(zip(similar['_ids'], similar['scores'])),
                **kwargs,
            )

    def _find_matches_then_similar(self, filter, *args, raw=False, features=None, **kwargs):

        only_like = self._test_only_like(filter)
        if not only_like:
            new_filter = self._remove_like_from_filter(filter)
            matches_cursor = SuperDuperCursor(
                self,
                new_filter,
                {'_id': 1},
                *args[1:],
                features=features,
                **kwargs,
            )
            ids = [x['_id'] for x in matches_cursor]
            similar = self._find_nearest(filter, ids=ids)
        else:  # pragma: no cover
            similar = self._find_nearest(filter)
        if raw:
            return Cursor(self, {'_id': {'$in': similar['_ids']}}, **kwargs) # pragma: no cover
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
        manifest = self.parent_if_appl[f'_models'].find_one({'name': name})
        if manifest is None:
            raise Exception(f'No such object of type "model", "{name}" has been registered.') # pragma: no cover
        if manifest['object'] is not None and not isinstance(manifest['object'], str):
            return self._load_object('models', name)
        elif isinstance(manifest['object'], str) and '.'  in manifest['object']:
            model_name, attr = manifest['object'].split('.')
            model = self._load_object('models', model_name)
            return getattr(model, attr)
        elif isinstance(manifest['object'], str):
            return self._load_object('models', manifest['object'])
        assert manifest.get('preprocessor') is not None \
            or manifest.get('forward') is not None
        preprocessor = None
        if manifest.get('preprocessor') is not None:
            preprocessor = self._load_object('preprocessors', manifest['preprocessor'])
        forward = None
        if manifest.get('forward') is not None:
            forward = self._load_object('forwards', manifest['forward'])
        postprocessor = None
        if manifest.get('postprocessor') is not None:
            postprocessor = self._load_object('postprocessors', manifest['postprocessor'])
        return create_container(preprocessor, forward, postprocessor)

    def _load_object(self, type, name):
        manifest = self.parent_if_appl[f'_{type}'].find_one({'name': name})
        if manifest is None:
            raise Exception(f'No such object of type "{type}", "{name}" has been registered.') # pragma: no cover
        m = self.parent_if_appl._load_pickled_file(manifest['object'])
        if type == 'models' and isinstance(m, torch.nn.Module):
            m.eval()
        return m

    def _load_hashes(self, name):
        info = self.parent_if_appl['_semantic_indexes'].find_one({'name': name})
        model_name = self.list_models(**{'active': True, 'name': {'$in': info['models']}})[0]
        model_info = self.parent_if_appl['_models'].find_one({'name': model_name})
        filter = model_info.get('filter', {})
        key = model_info.get('key', '_base')
        filter[f'_outputs.{key}.{model_name}'] = {'$exists': 1}
        n_docs = self.count_documents(filter)
        c = self.find(filter, {f'_outputs.{key}.{model_name}': 1})
        measure = self.measures[info['measure']]
        loaded = []
        ids = []
        docs = progressbar(c, total=n_docs)
        print(f'loading hashes: "{name}"')
        for r in docs:
            h = MongoStyleDict(r)[f'_outputs.{key}.{model_name}']
            loaded.append(h)
            ids.append(r['_id'])
        return hashes.HashSet(torch.stack(loaded), ids, measure=measure)

    def _load_pickled_file(self, file_id):
        return loading.load(file_id, filesystem=self.filesystem)

    def _get_dependencies_for_model(self, model):
        model_info = self['_models'].find_one({'name': model}, {'features': 1})
        if model_info is None:
            return []
        model_features = model_info.get('features', {})
        return list(model_features.values())

    def _get_model_for_neighbourhood(self, neigh):
        info = self['_neighbourhoods'].find_one({'name': neigh})
        si = self['_semantic_indexes'].find_one({'name': info['semantic_index']})
        return next(m for m in si['models'] if self['_models'].find_one({'name': m})['active'])

    def _submit_process_documents_with_model(self, model_name, sub_ids, dependencies, verbose=True):
        model_info = self.parent_if_appl['_models'].find_one({'name': model_name})
        if not self.remote:
            self._process_documents_with_model(
                model_name=model_name, ids=sub_ids, verbose=verbose,
                max_chunk_size=model_info.get('max_chunk_size', 5000),
                **model_info.get('loader_kwargs', {}),
            )
            if model_info.get('download', False):  # pragma: no cover
                self._download_content(ids=sub_ids)
        else:
            return superduper_requests.jobs.process(
                self.parent_if_appl.database.name,
                self.name,
                '_process_documents_with_model',
                model_name=model_name,
                ids=sub_ids,
                verbose=verbose,
                dependencies=dependencies,
                **model_info.get('loader_kwargs', {}),
            )

    def _create_filter_lookup(self, ids):
        filters = []
        for item in self.list_models(active=True):
            model_info = self.parent_if_appl['_models'].find_one({'name': item})
            filters.append(model_info.get('filter') or {})
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

    def _submit_download_content(self, ids):
        if not self.remote:
            print('downloading content from retrieved urls')
            self._download_content(ids=ids)
        else:
            return superduper_requests.jobs.process(
                self.database.name,
                self.name,
                '_download_content',
                ids=ids,
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
        if type_ == 'models':
            model_info = self.parent_if_appl['_models'].find_one({'name': item})
            if iteration == 0:
                dependencies = [download_id]
            else:
                model_dependencies = \
                    self.parent_if_appl._get_dependencies_for_model(item)
                dependencies = sum([
                    job_ids[('models', dep)]
                    for dep in model_dependencies
                ], [])
            filter_str = self._dict_to_str(model_info.get('filter') or {})
            sub_ids = lookup[filter_str]['_ids']
            process_id = \
                self._submit_process_documents_with_model(item, sub_ids, dependencies,
                                                          verbose=verbose)
            job_ids[(type_, item)].append(process_id)
            if model_info.get('download', False):  # pragma: no cover
                download_id = superduper_requests.jobs.process(
                    self.parent_if_appl.database.name,
                    self.name,
                    '_download_content',
                    ids=sub_ids,
                    dependencies=(process_id,),
                )
                job_ids[(type_, item)].append(download_id)
        elif type_ == 'neighbourhood':
            model = self.parent_if_appl._get_model_for_neighbourhood(item)
            dependencies = job_ids[('models', model)]
            process_id = self._submit_compute_with_neighbourhood(item, dependencies)
            job_ids[(type_, item)].append(process_id)
        return job_ids

    def _process_documents(self, ids, verbose=False):
        job_ids = defaultdict(lambda: [])
        download_id = self._submit_download_content(ids=ids)
        if not self.list_models(active=True):
            return
        lookup = self._create_filter_lookup(ids)
        G = self._create_plan()
        current = [('models', model) for model in self.list_models(active=True)
                   if not list(G.predecessors(('models', model)))]
        iteration = 0
        while current:
            for (type_, item) in current:
                job_ids = self._process_single_item(type_, item, iteration, lookup, job_ids, download_id,
                                                    verbose=verbose)
            current = sum([list(G.successors((type_, item))) for (type_, item) in current], [])
            iteration += 1
        return job_ids

    def _compute_model_outputs(self, ids, model_info, model=None, verbose=True):
        print('finding documents under filter')
        model_name = model_info['name']
        features = model_info.get('features', {})
        if features is None:
            features = {}  # pragma: no cover
        documents = list(self.find({'_id': {'$in': ids}}, features=features))
        ids = [r['_id'] for r in documents]  # find statement messes with the ordering
        for r in documents:
            del r['_id'] # _id can't be handled by dataloader
        print('done.')
        key = model_info.get('key', '_base')
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
        loader_kwargs = model_info.get('loader_kwargs', {})
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

    def _process_documents_with_model(self, model_name, ids, verbose=False,
                                      max_chunk_size=5000, num_workers=0, model=None):
        import sys
        sys.path.insert(0, os.getcwd())

        model_info = self.parent_if_appl['_models'].find_one({'name': model_name})
        if max_chunk_size is not None:
            for it, i in enumerate(range(0, len(ids), max_chunk_size)):
                print('computing chunk '
                      f'({it + 1}/{math.ceil(len(ids) / max_chunk_size)})')
                self._process_documents_with_model(
                    model_name,
                    ids=ids[i: i + max_chunk_size],
                    verbose=verbose,
                    num_workers=num_workers,
                    max_chunk_size=None,
                    model=model,
                )
            return

        outputs, ids = self._compute_model_outputs(ids, model_info, model=model, verbose=verbose)

        if model_info.get('type'):
            type = self.types[model_info['type']]
            outputs = [
                {
                    '_content': {
                        'bytes': type.encode(x),
                        'type': model_info['type']
                    }
                }
                for x in outputs
            ]

        self._write_model_outputs(outputs, ids, model_info)
        return outputs

    def _write_model_outputs(self, outputs, ids, model_info):
        key = model_info.get('key', '_base')
        model_name = model_info['name']
        print('bulk writing...')
        if 'target' not in model_info:
            self.bulk_write([
                UpdateOne({'_id': id},
                          {'$set': {f'_outputs.{key}.{model_name}': outputs[i]}})
                for i, id in enumerate(ids)
            ])
        else:  # pragma: no cover
            self.bulk_write([
                UpdateOne({'_id': id},
                          {'$set': {
                              model_info['target']: outputs[i]
                          }})
                for i, id in enumerate(ids)
            ])
        print('done.')

    @staticmethod
    def _remove_like_from_filter(r):
        return {k: v for k, v in r.items() if k != '$like'}

    def _replace_model(self, name, object):
        r = self['_models'].find_one({'name': name})
        assert name in self.list_models(), 'model doesn\'t exist to replace'
        if isinstance(r['object'], ObjectId):
            file_id = self._create_pickled_file(object)
            self.filesystem.delete(r['object'])
            self['_models'].update_one({'name': name}, {'$set': {'object': file_id}})
        elif isinstance(r['object'], str):
            self._replace_model(r['object'], object)
        else:
            assert r['object'] is None
            if isinstance(r['preprocessor'], str):
                file_id = self._create_pickled_file(object._preprocess)
                pre_info = self['_preprocessors'].find_one({'name': r['preprocessor']})
                self.filesystem.delete(pre_info['object'])
                self['_preprocessors'].update_one(
                    {'name': r['preprocessor']},
                    {'$set': {'object': file_id}}
                )

            if isinstance(r['forward'], str):
                file_id = self._create_pickled_file(object._forward)
                forward_info = self['_forwards'].find_one({'name': r['forward']})
                self.filesystem.delete(forward_info['object'])
                self['_forwards'].update_one({'name': r['forward']}, {'$set': {'object': file_id}})

            if isinstance(r['postprocessor'], str):
                file_id = self._create_pickled_file(object._postprocess)
                post_info = self['_postprocessors'].find_one({'name': r['postprocessor']})
                self.filesystem.delete(post_info['object'])
                self['_postprocessors'].update_one({'name': r['postprocessor']},
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

    @staticmethod
    def _test_only_like(r):  # pragma: no cover
        """
        >>> Collection._test_only_like({'$like': {'a': 'b'}})
        True
        >>> Collection._test_only_like({'a': {'$like': 'b'}})
        True
        >>> Collection._test_only_like({'a': {'$like': 'b'}, 'c': 2})
        False
        """
        if set(r.keys()) == {'$like'}:
            return True
        elif len(r.keys()) > 1:
            return False
        else:
            return Collection._test_only_like(next(iter(r.values())))

    def _train_imputation(self, name):

        import sys
        sys.path.insert(0, os.getcwd())

        if self.remote:
            return superduper_requests.jobs.process(self.database.name, self.name,
                                                    '_train_imputation', name)

        info = self['_imputations'].find_one({'name': name})
        self['_models'].update_one({'name': info['name']}, {'$set': {'training': True}})
        splitter = None
        if info.get('splitter'):
            splitter = self.splitters[info['splitter']]

        model = self.models[info['model']]
        target = self.models[info['target']]
        loss = self.losses[info['loss']]
        metrics = {k: self.metrics[k] for k in info['metrics']}

        model_info = self['_models'].find_one({'name': info['model']})
        target_info = self['_models'].find_one({'name': info['target']})

        keys = (model_info['key'], target_info['key'])

        training.train_tools.ImputationTrainer(
            name,
            cf['mongodb'],
            self.database.name,
            self.name,
            models=(model, target),
            keys=keys,
            model_names=(info['model'], info['target']),
            loss=loss,
            metrics=metrics,
            **info['trainer_kwargs'],
            save=self._replace_model,
            features=model_info.get('features', {}),
            filter=info['filter'],
            projection=info['projection'],
            splitter=splitter,
        ).train()

    def _train_semantic_index(self, name):

        import sys
        sys.path.insert(0, os.getcwd())

        if self.remote:
            return superduper_requests.jobs.process(self.database.name, self.name,
                                                    '_train_semantic_index', name)

        info = self['_semantic_indexes'].find_one({'name': name})
        model_names = info['models']
        models = []
        keys = []
        features = {}
        for save_name in model_names:
            r = self['_models'].find_one({'name': save_name})
            features.update(r.get('features', {}))
            keys.append(r['key'])
            models.append(self.models[save_name])

        metrics = {}
        for metric in info['metrics']:
            metrics[metric] = self.metrics[metric]

        loss = self.losses[info['loss']]
        splitter = info.get('splitter')
        if splitter:
            splitter = self.splitters[info['splitter']]

        active_model = self.list_models(**{'active': True, 'name': {'$in': model_names}})[0]
        filter_ = self['_models'].find_one({'name': active_model}, {'filter': 1})['filter']

        t = training.train_tools.SemanticIndexTrainer(
            name,
            cf['mongodb'],
            self.database.name,
            self.name,
            models=models,
            keys=keys,
            model_names=model_names,
            splitter=splitter,
            loss=loss,
            save=self._replace_model,
            watch='loss',
            metrics=metrics,
            features=features,
            validation_sets=info.get('validation_sets', ()),
            **info.get('trainer_kwargs', {}),
        )
        t.train()
