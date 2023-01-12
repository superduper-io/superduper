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

from superduperdb.cursor import SuperDuperCursor
from superduperdb.types.utils import convert_types

warnings.filterwarnings('ignore')
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

    - *create_imputation*
    - *create_loss*
    - *create_metric*
    - *create_model*
    - *create_neighbourhood*
    - *create_semantic_index*
    - *create_splitter*
    - *create_type*

    Deleting objects:

    - *delete_imputation*
    - *delete_loss*
    - *delete_metric*
    - *delete_model*
    - *delete_neighbourhood*
    - *delete_semantic_index*
    - *delete_splitter*
    - *delete_type*

    Accessing data:

    - *find_one*
    - *find*

    Inserting and updating data:

    - *insert_many*
    - *insert_one*
    - *replace_one*
    - *update_one*
    - *update_many*

    Viewing meta-data

    - *list_models*
    - *list_semantic_indexes*
    - *list_imputations*
    - *list_types*
    - *list_losss*
    - *list_metrics*
    - *list_types*

    Watching jobs

    - *watch_job*

    Key properties:

    - *hash_set* (in memory vectors for neighbourhood search)
    - *losses* (dictionary of losses)
    - *measures* (dictionary of measures)
    - *metrics* (dictionary of metrics)
    - *models* (dictionary of models)
    - *remote* (whether the client sends requests in thread or to a server)
    - *splitters* (dictionary of splitters)
    - *types* (dictionary of types)

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._hash_set = None
        self._all_hash_sets = ArgumentDefaultDict(self._load_hashes)
        self.remote = cf.get('remote', False)
        self._filesystem = None
        self._filesystem_name = f'_{self.database.name}:{self.name}:files'
        self._semantic_index = None

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

    def create_type(self, name, object):
        """
        Create a type directly from a Python object.

        :param name: name of type
        :param object: Python object
        """
        return self._create_object('types', name, object)

    def create_forward(self, name, object):
        """
        Create a forward directly from a Python object.

        :param name: name of forward
        :param object: Python object
        """
        return self._create_object('forwards', name, object)

    def create_imputation(self, name, model, loss, target, metrics=None, filter=None, projection=None,
                          n_epochs=20, **trainer_kwargs):
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
        :param n_epochs: Number of epochs to train
        :param trainer_kwargs: Keyword-arguments to forward to ``train_tools.ImputationTrainer``
        :return: job_ids of jobs required to create the imputation
        """

        created = {}
        if target is not None and isinstance(target, dict):  # pragma: no cover
            self.create_model(**target, active=False)
            target_name = target['name']
            created['target'] = True
        else: # pragma: no cover
            assert target in self.list_models()
            target_name = target
            created['target'] = False

        if isinstance(model, dict):  # pragma: no cover
            self.create_model(**model, process_docs=False)
            model_name = model['name']
            created['model'] = True
        else: # pragma: no cover
            assert model in self.list_models()
            model_name = model
            created['model'] = False

        if not isinstance(loss, str):  # pragma: no cover
            self.create_loss(**loss)
            loss = loss['name']
            created['loss'] = True
        else: # pragma: no cover
            assert loss in self.list_losses()
            created['loss'] = False

        created['metrics'] = []
        if metrics is not None:  # pragma: no cover
            for i, metric in enumerate(metrics):
                if isinstance(metric, str):
                    created['metrics'].append(False)
                if isinstance(metric, dict):
                    self.create_metric(metric['name'], metric['object'])
                    metrics[i] = metric['name']
                    created['metrics'].append(True)

        self['_imputations'].insert_one({
            'name': name,
            'model': model_name,
            'target': target_name,
            'metrics': metrics,
            'loss': loss,
            'projection': projection,
            'filter': filter,
            'n_epochs': n_epochs,
            'trainer_kwargs': trainer_kwargs,
            'created': created,
        })
        job_id = self._train_imputation(name)

        self['_models'].update_one({'name': model_name}, {'$set': {'training': False}})
        r = self['_models'].find_one({'name': model_name})
        ids = [r['_id'] for r in self.find(r.get('filter', {}), {'_id': 1})]

        if not self.remote:
            self._process_documents_with_model(model_name, ids, **r.get('loader_kwargs', {}),
                                               verbose=True)
        else:
            return [job_id, superduper_requests.jobs.process(
                self.database.name,
                self.name,
                '_process_documents_with_model',
                model_name=model_name,
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
                     key='_base', verbose=False, semantic_index=False,
                     process_docs=True, loader_kwargs=None, max_chunk_size=5000, features=None,
                     measure=None, neighbourboods=[]):
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
        created = {}

        object_should_exist = object is None and preprocessor is None and forward is None \
            and postprocessor is None

        if active and object is not None:
            assert hasattr(object, 'preprocess') or hasattr(object, 'forward')

        if not object_should_exist:
            assert '.' not in name, 'already existing model can only be used, if an attribute of that model is referred to'

        if object_should_exist:
            assert '.' in name, 'can only reference attribute of existing model'
            assert name.split('.')[0] in self.list_models()
        if semantic_index:
            assert measure is not None
            return self.create_semantic_index(
                name=name,
                models=[{
                    'name': name,
                    'object': object,
                    'preprocessor': preprocessor,
                    'forward': forward,
                    'postprocessor': postprocessor,
                    'filter': filter if filter else {},
                    'type': type,
                    'active': active,
                    'key': key,
                    'loader_kwargs': loader_kwargs if loader_kwargs else {},
                    'max_chunk_size': max_chunk_size,
                    'verbose': verbose,
                    'features': features if features else {},
                }],
                measure=measure,
            )
        assert name not in self['_models'].distinct('name'), \
            f'Model {name} already exists!'
        if type and isinstance(type, dict):  # pragma: no cover
            self.create_type(**type)
            type = type['name']
        elif type:  # pragma: no cover
            assert isinstance(type, str)
            assert type in self['_types'].distinct('name')

        if preprocessor is not None or forward is not None or postprocessor is not None:
            assert object is None
            if isinstance(preprocessor, str):
                preprocessor = self._preprocessors[preprocessor]
                created['preprocessor'] = False
            else:
                created['preprocessor']  = True
                self.create_preprocessor(**preprocessor)
                preprocessor = preprocessor['name']

            if isinstance(forward, str):
                forward = self._forwards[forward]
                created['forward'] = False
            else:
                created['forward'] = True
                self.create_forward(**forward)
                forward = forward['name']

            if isinstance(postprocessor, str):
                postprocessor = self._postprocessors[postprocessor]
                created['postprocessor'] = False
            else:
                created['postprocessor'] = True
                self.create_postprocessor(**postprocessor)
                postprocessor = postprocessor['name']

            file_id = None
        else:
            if object is not None:
                file_id = self._create_pickled_file(object)
            else:
                file_id = self['_models'].find_one({'name': name.split('.')[0]})['object']

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

    def create_semantic_index(self, name, models, measure, metrics=None, loss=None,
                              splitter=None, **trainer_kwargs):
        """
        :param name: Name of index
        :param models: List of existing models, or parameters to ``Collection.create_model``
        :param measure: Measure name, or parameters to ``Collection.create_measure``
        :param metrics: List of existing metrics, or parameters to ``Collection.create_metric``
        :param loss: Loss name, or parameters to ``Collection.create_loss``
        :param splitter: Splitter name, or parameters to ``Collection.create_splitter``
        :param trainer_kwargs: Keyword arguments to be passed to ``training.train_tools.RepresentationTrainer``
        :return: List of job identifiers if ``self.remote``
        """
        created = {'models': []}
        for i, man in enumerate(models):  # pragma: no cover
            if isinstance(man, str):
                assert man in self.list_models()
                created['models'].append(False)
                continue

            self.create_model(**man, process_docs=False)
            models[i] = man['name']
            created['models'].append(True)

        if metrics is not None:  # pragma: no cover
            created['metrics'] = []
            for i, man in enumerate(metrics):
                if isinstance(man, str):
                    assert man in self.list_metrics()
                    created['metrics'].append(False)
                    continue
                self.create_metric(**man)
                metrics[i] = man['name']
                created['metrics'].append(True)

        if splitter is not None:  # pragma: no cover
            if isinstance(splitter, str):
                assert splitter in self.list_splitters()
                created['splitter'] = False
            else:
                self.create_splitter(**splitter)
                splitter = splitter['name']
                created['splitter'] = True

        if isinstance(measure, str):  # pragma: no cover
            assert measure in self.list_measures()
            created['measure'] = False
        else:  # pragma: no cover
            self.create_measure(**measure)
            measure = measure['name']
            created['measure'] = True

        if loss is not None:  # pragma: no cover
            if len(models) == 1:
                assert splitter is not None, 'need a splitter for self-supervised ranking...'
            if isinstance(loss, str):
                assert loss in self.list_losses()
                created['loss'] = False
            else:
                self.create_loss(**loss)
                loss = loss['name']
                created['loss'] = True

        self['_semantic_indexes'].insert_one({
            'name': name,
            'models': models,
            'metrics': metrics,
            'loss': loss,
            'measure': measure,
            'splitter': splitter,
            'trainer_kwargs': trainer_kwargs,
            'created': created,
        })
        if loss is None:
            return

        active_model = self.list_models(**{'active': True, 'name': {'$in': models}})[0]
        filter_ = self['_models'].find_one({'name': active_model}, {'filter': 1})['filter']
        filter_['_fold'] = 'valid'
        self['_models'].update_one({'name': active_model}, {'$set': {'filter': filter_}})

        job_ids = []
        job_ids.append(self._train_semantic_index(name=name))

        active_models = self.list_models(**{'active': True, 'name': {'$in': models}})
        for model in active_models:
            model_info = self['_models'].find_one({'name': model})
            self['_models'].update_one({'name': model}, {'$set': {'filter': filter_}})
            ids = [x['_id'] for x in self.find(filter_, {'_id': 1})]
            if not self.remote:
                self._process_documents_with_model(
                    model,
                    ids,
                    verbose=True,
                    **model_info.get('loader_kwargs', {}),
                    max_chunk_size=model_info.get('max_chunk_size'),
                )
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

    def list_neighbourhoods(self):
        """
        List neighbourhoods.
        """
        return self['_neighbourhoods'].distinct('name')

    def list_jobs(self):
        """
        List jobs.
        """
        return self['_jobs'].distinct('identifier')

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

        info = self['_imputations'].find_one()

        self['_imputations'].delete_one({'name': name})

        for item in info['created']:
            if not info['created']:
                continue
            if item == 'model':
                self.delete_model(info['model'], force=True)
            if item == 'target':
                self.delete_model(info['target'], force=True)
            if item == 'loss':
                self.delete_loss(info['loss'], force=True)
            if item == 'metrics':
                for i, was_created in info['created']['metrics']:
                    if not was_created:
                        continue
                    self.delete_metric(info['metrics'][i], force=True)

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
        all_models = self.list_models()
        if '.' in name:  # pragma: no cover
            raise Exception('must delete parent model in order to delete <model>.<attribute>')
        models = [name]
        children = [y for y in all_models if y.startswith(f'{name}.')]
        models.extend(children)
        if not force: # pragma: no cover
            filters_ = []
            for m in models:
                filters_.append(self['_models'].find_one({'name': m})['filter'])
            n_documents = self.count_documents({'$and': filters_})
        do_delete = False
        if force or click.confirm(f'Are you sure you want to delete these models: {models}; '
                                  f'{n_documents} documents will be affected.',
                                  default=False):
            do_delete = True
        if not do_delete:
            return

        info = self['_models'].find_one({'name': name})
        if info['object'] is None:
            if info['preprocessor'] is not None:
                self._delete_objects('preprocessors', [info['preprocessor']], force=True)
            if info['forward'] is not None:
                self._delete_objects('forwards', [info['forward']], force=True)
            if info['postprocessor'] is not None:
                self._delete_objects('postprocessors', [info['postprocessor']], force=True)
            self['_models'].delete_one({'name': name})
            return

        deleted_info = [self['_models'].find_one({'name': m}) for m in models]
        _ = self._delete_objects(
            'models', objects=[x for x in models if '.' not in x], force=True
        )
        for r in deleted_info:
            print(f'unsetting output field _outputs.{r["key"]}.{r["name"]}')
            super().update_many(
                r.get('filter', {}),
                {'$unset': {f'_outputs.{r["key"]}.{r["name"]}': 1}}
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

    def delete_postprocess(self, postprocess, force=False):
        """
        Delete postprocess from collection

        :param name: Name of postprocess
        :param force: Toggle to ``True`` to skip confirmation
        """
        return self._delete_objects('postprocesss', objects=[postprocess], force=force)

    def delete_preprocess(self, preprocess, force=False):
        """
        Delete preprocess from collection

        :param name: Name of preprocess
        :param force: Toggle to ``True`` to skip confirmation
        """
        return self._delete_objects('preprocesss', objects=[preprocess], force=force)

    def delete_semantic_index(self, name, force=False):
        """
        Delete semantic index.

        :param name: Name of semantic index
        :param force: Toggle to ``True`` to skip confirmation
        """
        info = self['_semantic_indexes'].find_one({'name': name}, {'models': 1})
        if info is None:  # pragma: no cover
            return
        active_models = self.list_models(**{'active': True, 'name': {'$in': info['models']}})
        filters_ = []
        for m in active_models:
            filters_.append(self['_models'].find_one({'name': m}, {'filter': 1})['filter'])
        if not force: # pragma: no cover
            n_documents = self.count_documents({'$and': filters_})
        do_delete = False
        if force or click.confirm(f'Are you sure you want to delete this semantic index: {name}; '
                                  f'{n_documents} documents will be affected.'):
            do_delete = True

        if not do_delete:
            return

        for item in info['created']:
            if not info['created']:
                continue
            self['_semantic_indexes'].delete_one({'name': name})
            if item == 'models':
                for i, was_created in enumerate(info['created']['models']):
                    if not was_created:
                        continue
                    self.delete_model(info['models'][i], force=True)
            if item == 'loss':
                self.delete_loss(info['loss'], force=True)
            if item == 'metrics':
                for i, was_created in info['created']['metrics']:
                    if not was_created:
                        continue
                    self.delete_metric(info['metrics'][i], force=True)

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
        return self['_types'].distinct('name')

    def list_forwards(self):
        """
        List forwards
        :return: list of forwards
        """
        return self['_forwards'].distinct('name')

    def list_imputations(self):
        """
        List imputations
        :return: list of imputations
        """
        return self['_imputations'].distinct('name')

    def list_losses(self):
        """
        List losses
        :return: list of losses
        """
        return self['_losses'].distinct('name')

    def list_measures(self):
        """
        List measures
        :return: list of measures
        """
        return self['_measures'].distinct('name')

    def list_metrics(self):
        """
        List metrics
        :return: list of metrics
        """
        return self['_metrics'].distinct('name')

    def list_models(self, **kwargs):
        """
        List models
        :return: list of models
        """
        return self['_models'].distinct('name', kwargs)

    def list_postprocessors(self):
        """
        List postprocessors
        :return: list of postprocessors
        """
        return self['_postprocessors'].distinct('name')

    def list_preprocessors(self):
        """
        List preprocessors
        :return: list of preprocessors
        """
        return self['_preprocessors'].distinct('name')

    def list_semantic_indexes(self):
        """
        List semantic_indexes
        :return: list of semantic_indexes
        """
        return self['_semantic_indexes'].distinct('name')

    def list_splitters(self):
        """
        List splitters
        :return: list of splitters
        """
        return self['_splitters'].distinct('name')

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

    def _test_type(self, item):
        for c in self.list_types():
            if hasattr(self.types[c], 'isinstance'):
                if self.types[c].isinstance(item):
                    return c

    def _convert_types(self, r):
        for k in r:
            if isinstance(r[k], dict) and '_content' not in r[k]:
                r[k] = self._convert_types(r[k])
            c = self._test_type(r[k])
            if c is not None:
                r[k] = {'_content': {'bytes': self.types[c].encode(r[k]),
                                     'type': c}}
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
            filter = self._convert_types(filter)
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
            models = self['_semantic_indexes'].find_one({'name': self.semantic_index})['models']
            available_keys = list(filter['$like']['document'].keys()) + ['_base']
            man = self['_models'].find_one({
                'key': {'$in': available_keys},
                'name': {'$in': models},
            })
            model = self.models[man['name']]
            document = MongoStyleDict(filter['$like']['document'])
            if '_outputs' not in document:
                document['_outputs'] = {}
            info = self['_models'].find_one({'name': man['name']})
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

    def _load_model(self, name):
        manifest = self[f'_models'].find_one({'name': name})
        if manifest is None:
            raise Exception(f'No such object of type "model", "{name}" has been registered.') # pragma: no cover
        if manifest['object'] is not None:
            return self._load_object('models', name)
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
        manifest = self[f'_{type}'].find_one({'name': name})
        if manifest is None:
            raise Exception(f'No such object of type "{type}", "{name}" has been registered.') # pragma: no cover
        m = self._load_pickled_file(manifest['object'])
        if '.' in name:
            _, attribute = name.split('.')
            m = getattr(m, attribute)
        if type == 'models' and isinstance(m, torch.nn.Module):
            m.eval()
        return m

    def _load_hashes(self, name):
        info = self['_semantic_indexes'].find_one({'name': name})
        model_name = self.list_models(**{'active': True, 'name': {'$in': info['models']}})[0]
        model_info = self['_models'].find_one({'name': model_name})
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

    def _process_documents(self, ids, verbose=False):
        if not self.remote:
            self._download_content(ids=ids)
        else:
            job_ids = defaultdict(lambda: [])
            download_id = superduper_requests.jobs.process(
                self.database.name,
                self.name,
                '_download_content',
                ids=ids,
            )
        if not self.list_models(active=True):
            return         # pragma: no cover
        filters = []
        for item in self.list_models(active=True):
            model_info = self['_models'].find_one({'name': item})
            filters.append(model_info.get('filter', {}))
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

        G = self._create_plan()
        current = [('models', model) for model in self.list_models(active=True)
                   if not list(G.predecessors(('models', model)))]

        iteration = 0
        while current:
            for (type_, item) in current:
                if type_ == 'models':
                    model_info = self['_models'].find_one({'name': item})
                    filter_str = self._dict_to_str(model_info.get('filter', {}))
                    sub_ids = lookup[filter_str]['_ids']
                    if not sub_ids:  # pragma: no cover
                        continue
                    if not self.remote:
                        self._process_documents_with_model(
                            model_name=item, ids=sub_ids, verbose=verbose,
                            max_chunk_size=model_info.get('max_chunk_size', 5000),
                            **model_info.get('loader_kwargs', {}),
                        )
                        if model_info.get('download', False):  # pragma: no cover
                            self._download_content(ids=sub_ids)
                    else:
                        if iteration == 0:
                            dependencies = [download_id]
                        else:
                            model_dependencies = self._get_dependencies_for_model(item)
                            dependencies = sum([
                                job_ids[('models', dep)]
                                for dep in model_dependencies
                            ], [])
                        process_id = superduper_requests.jobs.process(
                            self.database.name,
                            self.name,
                            '_process_documents_with_model',
                            model_name=item,
                            ids=ids,
                            verbose=verbose,
                            dependencies=dependencies,
                            **model_info.get('loader_kwargs', {}),
                        )
                        job_ids[(type_, item)].append(process_id)
                        if model_info.get('download', False):  # pragma: no cover
                            download_id = superduper_requests.jobs.process(
                                self.database.name,
                                self.name,
                                '_download_content',
                                ids=sub_ids,
                                dependencies=(process_id,),
                            )
                elif type_ == 'neighbourhoods':
                    model = self._get_model_for_neighbourhood(item)
                    model_info = self['_models'].find_one({'name': model})
                    filter_str = self._dict_to_str(model_info.get('filter', {}))
                    sub_ids = lookup[filter_str]['_ids']
                    if not sub_ids:  # pragma: no cover
                        continue
                    if not self.remote:
                        self._compute_neighbourhood(item, sub_ids)
                    else:
                        process_id = superduper_requests.jobs.process(
                            self.database.name,
                            self.name,
                            '_compute_neighbourhood',
                            name=item,
                            ids=sub_ids,
                            dependencies=job_ids[('models', model)]
                        )
                        job_ids[(type_, item)].append(process_id)
                else:
                    raise NotImplementedError(f'unknown type_ {type_}')
            current = sum([list(G.successors((type_, item))) for (type_, item) in current], [])
            iteration += 1
        if self.remote:
            return dict(job_ids)

    def _process_documents_with_model(self, model_name, ids, batch_size=10, verbose=False,
                                      max_chunk_size=5000, num_workers=0, model=None):
        import sys
        sys.path.insert(0, os.getcwd())

        model_info = self['_models'].find_one({'name': model_name})
        if max_chunk_size is not None:
            for it, i in enumerate(range(0, len(ids), max_chunk_size)):
                print('computing chunk '
                      f'({it + 1}/{math.ceil(len(ids) / max_chunk_size)})')
                self._process_documents_with_model(
                    model_name,
                    ids=ids[i: i + max_chunk_size],
                    batch_size=batch_size,
                    verbose=verbose,
                    num_workers=num_workers,
                    max_chunk_size=None,
                    model=model,
                )
            return

        print('finding documents under filter')
        features = model_info.get('features', {})
        if features is None:
            features = {}    # pragma: no cover
        filter_ = model_info.get('filter', {})
        documents = list(self.find(filter_, features=features))
        ids = [r['_id'] for r in documents]
        for r in documents:
            del r['_id']
        print('done.')
        key = model_info.get('key', '_base')
        if key != '_base' or '_base' in features:
            passed_docs = [r[key] for r in documents]
        else:  # pragma: no cover
            passed_docs = documents
        if model is None:
            model = self.models[model_name]
        inputs = training.loading.BasicDataset(
            passed_docs,
            model.preprocess if hasattr(model, 'preprocess') else lambda x: x
        )
        loader_kwargs = model_info.get('loader_kwargs', {})
        if hasattr(model, 'forward'):
            if 'batch_size' not in loader_kwargs:
                loader_kwargs['batch_size'] = batch_size
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

        if model_info.get('type'):
            type = self.types[model_info['type']]
            tmp = [
                {model_name: {
                    '_content': {
                        'bytes': type.encode(x),
                        'type': model_info['type']
                    }
                }}
                for x in outputs
            ]
        else:
            tmp = [{model_name: out} for out in outputs]
        key = model_info.get('key', '_base')
        print('bulk writing...')
        if 'target' not in model_info:
            self.bulk_write([
                UpdateOne({'_id': id},
                          {'$set': {f'_outputs.{key}.{model_name}': tmp[i][model_name]}})
                for i, id in enumerate(ids)
            ])
        else:  # pragma: no cover
            self.bulk_write([
                UpdateOne({'_id': id},
                          {'$set': {
                              model_info['target']: tmp[i][model_name]
                          }})
                for i, id in enumerate(ids)
            ])
        print('done.')
        return tmp

    @staticmethod
    def _remove_like_from_filter(r):
        return {k: v for k, v in r.items() if k != '$like'}

    def _replace_model(self, name, object):
        r = self['_models'].find_one({'name': name})
        assert name in self.list_models(), 'model doesn\'t exist to replace'
        file_id = self._create_pickled_file(object)
        self.filesystem.delete(r['object'])
        self['_models'].update_one({'name': name}, {'$set': {'object': file_id}})

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
            save_names=(info['model'], info['target']),
            loss=loss,
            metrics=metrics,
            **info['trainer_kwargs'],
            save=self._replace_model,
            features=model_info.get('features', {}),
            filter=info['filter'],
            projection=info['projection'],
            n_epochs=info['n_epochs'],
        ).train()

    def _train_semantic_index(self, name):

        import sys
        sys.path.insert(0, os.getcwd())

        if self.remote:
            return superduper_requests.jobs.process(self.database.name, self.name,
                                                    '_train_semantic_index', name)

        info = self['_semantic_indexes'].find_one({'name': name})
        save_names = info['models']
        models = []
        keys = []
        features = {}
        for save_name in save_names:
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

        active_model = self.list_models(**{'active': True, 'name': {'$in': save_names}})[0]
        filter_ = self['_models'].find_one({'name': active_model}, {'filter': 1})['filter']
        if splitter is not None:
            filter_['_fold'] = 'temp'
        else:
            filter_['_fold'] = 'valid'

        t = training.train_tools.RepresentationTrainer(
            name,
            cf['mongodb'],
            self.database.name,
            self.name,
            models=models,
            keys=keys,
            save_names=save_names,
            splitter=splitter,
            loss=loss,
            save=self._replace_model,
            watch='loss',
            metrics=metrics,
            features=features,
            **info.get('trainer_kwargs', {}),
        )

        if splitter is not None:
            filter_['_training_id'] = t.training_id
            self['_models'].update_one({'name': active_model}, {'$set': {'filter': filter_}})

        t.train()

        del filter_['_fold']
        # TODO create training_id the moment the semantic index function is called not down here
        if '_training_id' in filter_:
            del filter_['_training_id']

        self['_models'].update_one({'name': active_model}, {'$set': {'filter': filter_}})

