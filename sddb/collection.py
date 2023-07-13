import time

import click
from collections import defaultdict
import copy
import hashlib
import multiprocessing

import gridfs
import math
import networkx
import random
import warnings

warnings.filterwarnings('ignore')

from pymongo import UpdateOne
from pymongo.collection import Collection as BaseCollection
from pymongo.cursor import Cursor
import torch.utils.data

from sddb import cf
from sddb.lookup import hashes
from sddb.models import loading
from sddb import requests as sddb_requests
from sddb import training
from sddb.utils import apply_model, unpack_batch, MongoStyleDict, Downloader, Progress


class ArgumentDefaultDict(defaultdict):
    def __getitem__(self, item):
        if item not in self.keys():
            self[item] = self.default_factory(item)
        return super().__getitem__(item)


def convert_types(r, convert=True, converters=None):
    if converters is None:
        converters = {}
    for k in r:
        if isinstance(r[k], dict):
            if '_content' in r[k]:
                if 'bytes' in r[k]['_content']:
                    if convert:
                        converter = converters[r[k]['_content']['converter']]
                        r[k] = converter.decode(r[k]['_content']['bytes'])
                    else:
                        pass
                elif 'path' in r[k]['_content']:
                    try:
                        with open(r[k]['_content']['path'], 'rb') as f:
                            if convert:
                                converter = converters[r[k]['_content']['converter']]
                                r[k] = converter.decode(f.read())
                            else:
                                r[k]['_content']['bytes'] = f.read()
                    except FileNotFoundError:
                        return
                else:
                    raise NotImplementedError(
                        f'neither "bytes" nor "path" found in record {r}'
                    )
            else:
                convert_types(r[k], convert=convert, converters=converters)
    return r


class SddbCursor(Cursor):
    def __init__(self, collection, *args, features=None, convert=True, scores=None,
                 similar_join=None, **kwargs):
        """
        Cursor subclassing *pymongo.cursor.Cursor*. Converts content on disk to Python objects
        if *convert==True*. If *features* are specified, these are substituted in the records
        for the raw data. This is useful, for instance if images are present, and they should
        be featurized by a certain model. If *scores* are added, these are added to the results
        records.

        :param collection: collection
        :param *args: args to pass to super()
        :param features: feature dictionary
        :param convert: toggle to *True* to interpret bytes in records using converters
        :param scores: similarity scores to add to records
        :param similar_join: replace ids by documents in subfield of _like
        :param **kwargs: kwargs to pass to super()
        """
        super().__init__(collection, *args, **kwargs)
        self.attr_collection = collection
        self.features = features
        self.convert = convert
        self.scores = scores
        self.similar_join = similar_join
        self._args = args
        self._kwargs = kwargs

    def get_ids(self, r):
        return r['_like'][self.similar_join]

    def next(self):
        r = super().next()
        if self.scores is not None:
            r['_score'] = self.scores[r['_id']]
        if self.features is not None and self.features:
            r = MongoStyleDict(r)
            for k in self.features:
                r[k] =  r['_outputs'][k][self.features[k]]
        if self.similar_join is not None:
            if self.similar_join in r.get('_like', {}):
                ids = r['_like'][self.similar_join]
                lookup = {r['_id']: r for r in self.collection.find({'_id': {'$in': ids}},
                                                                    *self._args[1:],
                                                                    **self._kwargs)}
                for i, id_ in enumerate(r['_like'][self.similar_join]):
                    r['_like'][self.similar_join][i] = lookup[id_]

        r = convert_types(r, convert=self.convert, converters=self.attr_collection.converters)
        if r is None:
            return self.next()
        else:
            return r

    __next__ = next


class Collection(BaseCollection):
    """
    SuperDuperDB collection type, subclassing *pymongo.collection.Collection*.
    Key methods are:

    Creating objects:

    - *create_model*
    - *create_semantic_index*
    - *create_imputation*
    - *create_converter*
    - *create_loss*
    - *create_metric*

    Inserting, searching and updating data:

    - *insert_one
    - *find_one*
    - *update_one*
    - *insert_many
    - *find_many*
    - *update_many*

    Viewing meta-data

    - *list_models*
    - *list_semantic_indexes*
    - *list_imputations*
    - *list_converters*
    - *list_losss*
    - *list_metrics*

    Key properties:

    - *meta* (meta data of collection)
    - *models* (dictionary of models)
    - *losses* (dictionary of losses)
    - *metrics* (dictionary of metrics)
    - *converters* (dictionary of converters)

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._meta = None
        self._semantic_index = self.meta.get('semantic_index')
        self._semantic_index_data = self['_semantic_indexes'].find_one({
            'name': self._semantic_index
        })
        self._hash_set = None
        self._model_info = ArgumentDefaultDict(self._get_model_info)
        self._all_hash_sets = ArgumentDefaultDict(self._load_hashes)
        self.single_thread = cf.get('single_thread', True)
        self.remote = cf.get('remote', False)
        self.download_timeout = None
        self._filesystem = None
        self._filesystem_name = f'_{self.database.name}:{self.name}:files'

        self.models = ArgumentDefaultDict(lambda x: self._load_object('models', x))
        self.converters = ArgumentDefaultDict(lambda x: self._load_object('converters', x))
        self.losses = ArgumentDefaultDict(lambda x: self._load_object('losses', x))
        self.metrics = ArgumentDefaultDict(lambda x: self._load_object('metrics', x))
        self.measures = ArgumentDefaultDict(lambda x: self._load_object('measures', x))
        self.splitters = ArgumentDefaultDict(lambda x: self._load_object('splitters', x))

        self.headers = None

    def random_id(self, filter_=None):
        if filter_ is None:
            filter_ = {}
        for r in self.aggregate([
            {'$match': filter_},
            {'$sample': {'size': 1}},
            {'$project': {'_id': 1}}
        ]):
            return r['_id']

    @property
    def filesystem(self):
        if self._filesystem is None:
            self._filesystem = gridfs.GridFS(
                self.database.client[self._filesystem_name]
            )
        return self._filesystem

    def _create_pickled_file(self, object):
        return loading.save(object, filesystem=self.filesystem)

    def _load_pickled_file(self, file_id):
        return loading.load(file_id, filesystem=self.filesystem)

    def _create_object(self, type, name, object):
        assert name not in self[f'_{type}'].distinct('name')
        file_id = self._create_pickled_file(object)
        self[f'_{type}'].insert_one({'name': name, 'object': file_id})

    def create_loss(self, name, object):
        """
        Create a loss function directly from a Python object.

        :param name: name of loss function
        :param object: Python object
        """
        return self._create_object('losses', name, object)

    def create_model(self, name, object=None, filter=None, converter=None, active=True, in_memory=False,
                     dependencies=None, key='_base', verbose=False, semantic_index=False,
                     process_docs=True, loader_kwargs=None, max_chunk_size=5000, features=None,
                     measure=None):
        """
        Create a model registered in the collection directly from a python session.
        The added model will then watch incoming records and add outputs computed on those
        records into the "_outputs" of the records. The model is then stored inside MongoDB and can
        be accessed using the *SddbClient*.

        :param name: name of model
        :param object: if specified the model object (pickle-able) else None if model already exists
        :param filter: filter specifying which documents model acts on
        :param converter: converter for converting model outputs back and forth from bytes
        :param active: toggle to *False* if model should not actively process incoming data
        :param in_memory: toggle to *True* if model should only exist in current session
        :param dependencies: list of models on which model's inputs depend (executed prior)
        :param key: key in records on which model acts (default whole record "_base")
        :param verbose: toggle to *True* if processing on data is verbose
        :param semantic_index: toggle to *True*
        :param process_docs: toggle to *False* if documents not to be processed by models
        :param loader_kwargs: kwargs to be passed to dataloader for model in processing
        :param max_chunk_size: maximum chunk size of documents to be held in memory simultaneously
        :param features: dictionary of features to be substituted from model outputs to record
        """
        if loader_kwargs is None:
            loader_kwargs = {}

        if active and object is not None:
            assert hasattr(object, 'preprocess') or hasattr(object, 'forward')
        if object is None or '.' in name:
            assert object is None, 'if attribute of model used, can\'t respecify object'
            assert '.' in name, 'already existing model can only be used, if an attribute of that model is referred to'
            assert name.split('.')[0] in self.list_models()
        if semantic_index:
            assert measure is not None
            return self.create_semantic_index(
                name=name,
                models=[{
                    'name': name,
                    'object': object,
                    'filter': filter,
                    'converter': converter,
                    'active': active,
                    'dependencies': dependencies,
                    'key': key,
                    'loader_kwargs': loader_kwargs,
                    'max_chunk_size': max_chunk_size,
                    'verbose': verbose,
                    'features': features,
                }],
                measure=measure,
            )
        if dependencies is None:
            dependencies = []
        assert name not in self['_models'].distinct('name'), \
            f'Model {name} already exists!'
        if converter and isinstance(converter, dict):
            self.create_converter(**converter)
            converter = converter['name']
        elif converter:
            assert isinstance(converter, str)
            assert converter in self['_converters'].distinct('name')
        if not in_memory:
            if object is not None:
                file_id = self._create_pickled_file(object)
            else:
                file_id = self['_models'].find_one({'name': name.split('.')[0]})['object']
            self['_models'].insert_one({
                'name': name,
                'object': file_id,
                'filter': filter if filter else {},
                'converter': converter,
                'active': active,
                'dependencies': dependencies,
                'key': key,
                'loader_kwargs': loader_kwargs,
                'max_chunk_size': max_chunk_size,
                'features': features,
                'training': False,
            })
        else:
            self.models[name] = object

        if process_docs and active:
            ids = [r['_id'] for r in self.find(filter if filter else {}, {'_id': 1})]
            if ids:
                if self.single_thread:
                    self.process_documents_with_model(name, ids, verbose=verbose,
                                                      max_chunk_size=max_chunk_size)
                else:
                    return sddb_requests.jobs.process(
                        self.database.name,
                        self.name,
                        'process_documents_with_model',
                        name,
                        ids,
                        verbose=verbose,
                        max_chunk_size=max_chunk_size
                    )

    def watch_job(self, identifier):
        try:
            status = 'running'
            n_lines = 0
            while status == 'running':
                r = self['_jobs'].find_one({'identifier': identifier}, {'stdout': 1, 'status': 1})
                if len(r['stdout']) > n_lines:
                    print(''.join(r['stdout'][n_lines:]), end='')
                n_lines = len(r['stdout'])
                status = r['status']
                time.sleep(0.2)
            if status == 'failed':
                print(r['msg'])
        except KeyboardInterrupt:
            return

    def _load_object(self, type, name):
        manifest = self[f'_{type}'].find_one({'name': name})
        if manifest is None:
            raise Exception(f'No such object of type "{type}", "{name}" has been registered.')
        m = self._load_pickled_file(manifest['object'])
        if '.' in name:
            _, attribute = name.split('.')
            m = getattr(m, attribute)
        if type == 'models' and hasattr(m, 'eval'):
            m.eval()
        return m

    def _get_meta(self):
        m = self['_meta'].find_one()
        if m is None:
            return {}
        return m

    def _get_model_info(self, name):
        return self['_models'].find_one({'name': name})

    @property
    def active_models(self):
        return [self._model_info[x]['name'] for x in self.list_models()
                if self._model_info[x].get('active', True)]

    @active_models.setter
    def active_models(self, value):
        for x in self.list_models():
            if x in value:
                self['_models'].update_one({'name': x}, {'$set': {'active': True}})
            else:
                self['_models'].update_one({'name': x}, {'$set': {'active': False}})
        self._model_info = ArgumentDefaultDict(self._get_model_info)

    @property
    def meta(self):
        return self._get_meta()

    @property
    def semantic_index_name(self):
        return (
            self._semantic_index
            if self._semantic_index is not None
            else self.meta['semantic_index']
        )

    @property
    def semantic_index(self):
        if self._semantic_index_data is None:
            self._semantic_index_data = self['_semantic_indexes'].find_one(
                {'name': self.meta['semantic_index']}
            )
        for i, r_m in enumerate(self._semantic_index_data['models']):
            if isinstance(r_m, str):
                self._semantic_index_data['models'][i] = self._model_info[r_m]
        self._semantic_index_data['active_models'] = [
            x['name'] for x in self._semantic_index_data['models'] if x['active']
        ]
        return self._semantic_index_data

    @semantic_index.setter
    def semantic_index(self, value):
        assert value in self.list_semantic_indexes()
        self._semantic_index = value
        self._semantic_index_data = self['_semantic_indexes'].find_one({'name': value})

    @staticmethod
    def _gather_urls_for_document(r):
        '''
        >>> Collection._gather_urls_for_document({'a': {'_content': {'url': 'test'}}})
        (['test'], ['a'])
        >>> d = {'b': {'a': {'_content': {'url': 'test'}}}}
        >>> Collection._gather_urls_for_document(d)
        (['test'], ['b.a'])
        '''
        urls = []
        keys = []
        for k in r:
            if isinstance(r[k], dict) and '_content' in r[k]:
                if 'url' in r[k]['_content'] and 'path' not in r[k]['_content']:
                    keys.append(k)
                    urls.append(r[k]['_content']['url'])
            elif isinstance(r[k], dict) and '_content' not in r[k]:
                sub_urls, sub_keys = Collection._gather_urls_for_document(r[k])
                urls.extend(sub_urls)
                keys.extend([f'{k}.{key}' for key in sub_keys])
        return urls, keys

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

    def _load_hashes(self, name):
        info = self['_semantic_indexes'].find_one({'name': name})
        model_name = next(x for x in info['models'] if self._model_info[x]['active'])
        filter = self._model_info[model_name].get('filter', {})
        key = self._model_info[model_name].get('key', '_base')
        filter[f'_outputs.{key}.{model_name}'] = {'$exists': 1}
        n_docs = self.count_documents(filter)
        c = self.find(filter, {f'_outputs.{key}.{model_name}': 1})
        measure = self.measures[info['measure']]
        loaded = []
        ids = []
        docs = Progress()(c, total=n_docs)
        print(f'loading hashes: "{name}"')
        for r in docs:
            h = MongoStyleDict(r)[f'_outputs.{key}.{model_name}']
            loaded.append(h)
            ids.append(r['_id'])
        return hashes.HashSet(torch.stack(loaded), ids, measure=measure)

    @property
    def hash_set(self):
        if self.semantic_index_name is None:
            raise Exception('No semantic index has been set!')
        return self._all_hash_sets[self.semantic_index_name]

    def unset_hash_set(self):
        if self.semantic_index_name in self._all_hash_sets:
            del self._all_hash_sets[self.semantic_index_name]

    def refresh_model(self, name, verbose=False):
        info = self._model_info[name]
        ids = [r['_id'] for r in self.find(info.get('filter', {}), {'_id': 1})]
        self.process_documents_with_model(name,
                                          ids=ids,
                                          **info['loader_kwargs'],
                                          max_chunk_size=info.get('max_chunk_size'),
                                          verbose=verbose)

    def process_documents_with_model(self, model_name, ids, batch_size=10,
                                     verbose=False, max_chunk_size=5000, num_workers=0, model=None):
        model_info = self._model_info[model_name]
        if max_chunk_size is not None:
            for it, i in enumerate(range(0, len(ids), max_chunk_size)):
                print('computing chunk '
                      f'({it + 1}/{math.ceil(len(ids) / max_chunk_size)})')
                self.process_documents_with_model(
                    model_name,
                    ids=ids[i: i + max_chunk_size],
                    batch_size=batch_size,
                    verbose=verbose,
                    num_workers=num_workers,
                    max_chunk_size=None,
                )
            return
        print('getting requires')
        if 'requires' not in self._model_info[model_name]:
            filter = {'_id': {'$in': ids}}
        else:
            filter = {'_id': {'$in': ids},
                      self._model_info[model_name]['requires']: {'$exists': 1}}

        print('finding documents under filter')
        features = self._model_info[model_name].get('features', {})
        if features is None:
            features = {}
        documents = list(self.find(filter, features=features))
        ids = [r['_id'] for r in documents]
        for r in documents:
            del r['_id']
        print('done.')
        key = model_info.get('key', '_base')
        if key != '_base' or '_base' in features:
            passed_docs = [r[key] for r in documents]
        else:
            passed_docs = documents
        if model is None:
            model = self.models[model_name]
        inputs = training.loading.BasicDataset(
            passed_docs,
            model.preprocess if hasattr(model, 'preprocess') else lambda x: x
        )
        if hasattr(model, 'forward'):
            loader_kwargs = self._model_info[model_name].get('loader_kwargs', {})
            if 'batch_size' not in loader_kwargs:
                loader_kwargs['batch_size'] = batch_size
            loader = torch.utils.data.DataLoader(
                inputs,
                **loader_kwargs,
            )
            if verbose:
                print(f'processing with {model_name}')
                loader = Progress()(loader)
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
            n_workers = self.meta.get('n_workers', 0)
            outputs = []
            if n_workers:
                pool = multiprocessing.Pool(processes=n_workers)
                for r in pool.map(model.preprocess, passed_docs):
                    outputs.append(r)
                pool.close()
                pool.join()
            else:
                for r in passed_docs:
                    outputs.append(model.preprocess(r))

        if self._model_info[model_name].get('converter'):
            converter = self.converters[self._model_info[model_name]['converter']]
            tmp = [
                {model_name: {
                    '_content': {
                        'bytes': converter.encode(x),
                        'converter': self._model_info[model_name]['converter']
                    }
                }}
                for x in outputs
            ]
        else:
            tmp = [{model_name: out} for out in outputs]
        key = self._model_info[model_name].get('key', '_base')
        print('bulk writing...')
        if 'target' not in self._model_info[model_name]:
            self.bulk_write([
                UpdateOne({'_id': id},
                          {'$set': {f'_outputs.{key}.{model_name}': tmp[i][model_name]}})
                for i, id in enumerate(ids)
            ])
        else:
            self.bulk_write([
                UpdateOne({'_id': id},
                          {'$set': {
                              self._model_info[model_name]['target']: tmp[i][model_name]
                          }})
                for i, id in enumerate(ids)
            ])
        print('done.')
        return tmp

    def _compute_neighbourhood(self, name, ids):
        info = self['_neighbourhoods'].find_one({'name': name})
        h = self._all_hash_sets[info['name']]
        print(f'computing neighbours based on neighbour "{name}" and '
              f'index "{info["semantic_index"]}"')
        for i in Progress()(range(0, len(ids), info['batch_size'])):
            sub = ids[i: i + info['batch_size']]
            results = h.find_nearest_from_ids(sub, n=info['n'])
            similar_ids = [res['ids'] for res in results]
            self.bulk_write([
                UpdateOne({'_id': id_}, {'$set': {f'_like.{name}': sids}})
                for id_, sids in zip(sub, similar_ids)
            ])

    def list_neighbourhoods(self):
        return self['_neighbourhoods'].distinct('name')

    def list_jobs(self):
        return self['_jobs'].distinct('identifier')

    def create_neighbourhood(self, name, n=10, semantic_index=None, batch_size=100):
        assert name not in self.list_neighbourhoods()
        if semantic_index is None:
            assert self.semantic_index_name
        self['_neighbourhoods'].insert_one({
            'name': name,
            'semantic_index': semantic_index if semantic_index else self.semantic_index_name,
            'n': n,
            'batch_size': batch_size,
        })
        info = self['_semantic_indexes'].find_one({'name': name})
        model_name = next(m for m in info['models'] if self._model_info[m]['active'])
        filter_ = self._model_info[model_name].get('filter', {})
        ids = [r['_id'] for r in self.find(filter_, {'_id': 1})]
        if self.single_thread:
            self._compute_neighbourhood(name, ids)
        else:
            raise NotImplementedError

    @staticmethod
    def standardize_dict(d):
        keys = sorted(list(d.keys()))
        out = {}
        for k in keys:
            if isinstance(d[k], dict):
                out[k] = Collection.standardize_dict(d[k])
            else:
                out[k] = d[k]
        return out

    @staticmethod
    def dict_to_str(d):
        sd = Collection.standardize_dict(d)
        return str(sd)

    def _process_documents(self, ids, verbose=False, blocking=False):
        if self.single_thread:
            self._download_content(ids=ids)
        else:
            job_ids = defaultdict(lambda: [])
            download_id = sddb_requests.jobs.process(
                self.database.name,
                self.name,
                'download_content',
                ids=ids,
            )
        if not self.acti:
            return
        filters = []
        for model in self.active_models:
            filters.append(self._model_info[model].get('filter', {}))
        filter_lookup = {self.dict_to_str(f): f for f in filters}
        lookup = {}
        for filter_str in filter_lookup:
            if filter_str not in lookup:
                tmp_ids = [
                    r['_id']
                    for r in super().find({
                        '$and': [{'_id': {'$in': ids}}, filter_lookup[filter_str]]
                    })
                ]
                lookup[filter_str] = {'ids': tmp_ids}

        G = self._create_plan()
        current = [model for model in self.active_models if not list(G.predecessors(model))]
        iteration = 0
        while current:
            for model in current:
                filter = self._model_info[model].get('filter', {})
                filter_str = self.dict_to_str(filter)
                sub_ids = lookup[filter_str]['ids']
                if not sub_ids:
                    continue

                if self.single_thread:
                    self.process_documents_with_model(
                        model_name=model, ids=sub_ids, verbose=verbose,
                        max_chunk_size=self._model_info[model].get('max_chunk_size', 5000),
                        **self._model_info[model].get('loader_kwargs', {}),
                    )
                    if self._model_info[model].get('download', False):
                        self._download_content(ids=sub_ids)
                else:
                    if iteration == 0:
                        dependencies = [download_id]
                    else:
                        dependencies = sum([
                            job_ids[dep]
                            for dep in self._model_info[model].get('dependencies', [])
                        ], [])
                    process_id = sddb_requests.jobs.process(
                        self.database.name,
                        self.name,
                        'process_documents_with_model',
                        model_name=model,
                        ids=ids,
                        verbose=verbose,
                        blocking=blocking,
                        dependencies=dependencies,
                        **self._model_info[model].get('loader_kwargs', {}),
                    )
                    job_ids[model].append(process_id)
                    if self._model_info[model].get('download', False):
                        download_id = sddb_requests.jobs.process(
                            self.database.name,
                            self.name,
                            'download_content',
                            ids=sub_ids,
                            dependencies=(process_id,),
                        )
            current = sum([list(G.successors(model)) for model in current], [])
            iteration += 1

    def _create_plan(self):
        G = networkx.DiGraph()
        for model in self.active_models:
            G.add_node(model)
        for model in self.list_models():
            for dep in self._model_info[model].get('dependencies', ()):
                G.add_edge(dep, model)
        assert networkx.is_directed_acyclic_graph(G)
        return G

    def insert_one(
        self,
        document,
        *args,
        **kwargs,
    ):
        """
        Insert a document into database.

        :param documents: list of documents
        :param *args: args to be passed to super()
        :param **kwargs: kwargs to be passed to super()
        """
        if 'valid_probability' in self.meta:
            r = random.random()
            document['_fold'] = (
                'valid' if r < self.meta['valid_probability'] else 'train'
            )
        output = super().insert_one(document, *args, **kwargs)
        if self.list_models():
            self._process_documents([output.inserted_id],
                                    blocking=True)
        return output

    def _download_content(self, ids=None, documents=None, download_folder=None,
                          timeout=-1, raises=True):
        if documents is None:
            assert ids is not None
            documents = list(self.find({'_id': {'$in': ids}}, {'_outputs': 0}, raw=True))
        urls, keys, place_ids = self._gather_urls(documents)
        if not urls:
            return
        files = []
        if download_folder is None:
            download_folder = self.meta.get('downloads', 'data/downloads')
        for url in urls:
            files.append(
                f'{download_folder}/{hashlib.sha1(url.encode("utf-8")).hexdigest()}'
            )
        downloader = Downloader(
            urls=urls,
            files=files,
            n_workers=self.meta.get('n_download_workers', 0),
            timeout=self.download_timeout if timeout == -1  else timeout,
            headers=self.headers,
            raises=raises,
        )
        downloader.go()
        if ids is not None:
            self.bulk_write([
                UpdateOne({'_id': id_}, {'$set': {f'{key}._content.path': file}})
                for id_, key, file in zip(place_ids, keys, files)
            ])
        else:
            for id_, key, file in zip(place_ids, keys, files):
                tmp = MongoStyleDict(documents[id_])
                tmp[f'{key}._content.path'] = file
                documents[id_] = tmp
            return documents

    def insert_many(
        self,
        documents,
        *args,
        verbose=True,
        **kwargs,
    ):
        """
        Insert many documents into database.

        :param documents: list of documents
        :param *args: args to be passed to super()
        :param verbose: toggle to *True* to display outputs during computation
        :param **kwargs: kwargs to be passed to super()
        """
        for document in documents:
            r = random.random()
            document['_fold'] = 'valid' if r < self.meta.get('valid_probability', 0.05) else 'train'
        output = super().insert_many(documents, *args, **kwargs)
        self._process_documents(output.inserted_ids, verbose=verbose)
        return output

    def update_one(
        self,
        filter,
        *args,
        refresh=True,
        **kwargs,
    ):
        """
        Update a document in the database.

        :param filter: MongoDB like filter
        :param *args: args to be passed to super()
        :param refresh: Toggle to *False* to not process document again with models.
        :param **kwargs: kwargs to be passed to super()
        """
        if refresh and self.list_models():
            id_ = super().find_one(filter, {'_id': 1})['_id']
        result = super().update_one(filter, *args, **kwargs)
        if refresh and self.list_models():
            document = super().find_one({'_id': id_}, {'_outputs': 0})
            self._process_documents([document['_id']])
        return result

    def update_many(
        self,
        filter,
        *args,
        refresh=True,
        **kwargs,
    ):
        if refresh and self.list_models():
            ids = [r['_id'] for r in super().find(filter, {'_id': 1})]
        result = super().update_many(filter, *args, **kwargs)
        if refresh and self.list_models():
            self._process_documents(ids)
        return result

    def _find_nearest(self, filter, ids=None):
        assert '$like' in filter
        if set(filter['$like'].keys()) == {'_id'}:
            filter['$like'] = {'document': {'_id': filter['$like']['_id']}, 'n': 10}
        if ids is None:
            hash_set = self.hash_set
        else:
            hash_set = self.hash_set[ids]

        if '_id' in filter['$like']['document']:
            return hash_set.find_nearest_from_id(filter['$like']['document']['_id'],
                                                 n=filter['$like']['n'])
        else:
            try:
                man = next(man for man in self.semantic_index['models']
                           if man['key'] in filter['$like']['document'])
            except StopIteration:
                man = next(man for man in self.semantic_index['models']
                           if man['key'] == '_base')
            model = self.models[man['name']]
            document = MongoStyleDict(filter['$like']['document'])
            if '_outputs' not in document:
                document['_outputs'] = {}
            features = self._model_info[man['name']].get('features', {})
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

    def _cleanup_failed_semantic_index(self):
        raise NotImplementedError

    def find_one(self, filter=None, *args, similar_first=False, raw=False, features=None,
                 convert=True, download=False, similar_join=None, **kwargs):
        """
        Behaves like MongoDB *find_one* with exception of "$like" operator.
        See *Collection.find* for more details.

        :param filter: filter dictionary
        :param *args: args passed to super()
        :param similar_first: toggle to *True* to first find similar things, and then
                              apply filter to these things
        :param raw: toggle to *True* to not convert bytes to Python objects but return raw bytes
        :param features: dictionary of model outputs to replace for dictionary elements
        :param convert: (TODO remove) convert cursor byte fields to Python types
        :param download: toggle to *True* in case query has downloadable "_content" components
        :param similar_join: replace ids by documents
        :param **kwargs: kwargs to be passed to super()
        """
        if self.remote:
            return sddb_requests.client.find_one(
                self.database.name,
                self.name,
                filter,
                *args,
                similar_first=similar_first,
                raw=raw,
                features=features,
                download=download,
                **kwargs,
            )
        cursor = self.find(filter, *args, raw=raw, features=features, convert=convert,
                           download=download, similar_join=similar_join, **kwargs)
        for result in cursor.limit(-1):
            return result
        return None

    @staticmethod
    def _test_only_like(r):
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

    @staticmethod
    def _remove_like_from_filter(r):
        return {k: v for k, v in r.items() if k != '$like'}

    @staticmethod
    def _find_like_operator(r):
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

    def _find_similar_then_matches(self, filter, *args, raw=False, convert=True, features=None,
                                   **kwargs):

        similar = self._find_nearest(filter)
        only_like = self._test_only_like(filter)
        if not only_like:
            new_filter = self._remove_like_from_filter(filter)
            filter = {
                '$and': [
                    new_filter,
                    {'_id': {'$in': similar['ids']}}
                ]
            }
        else:
            filter = {'_id': {'$in': similar['ids']}}
        if raw:
            return Cursor(self, filter, *args, **kwargs)
        else:
            return SddbCursor(
                self,
                filter,
                *args,
                convert=convert,
                features=features,
                scores=dict(zip(similar['ids'], similar['scores'])),
                **kwargs,
            )

    def _find_matches_then_similar(self, filter, *args, raw=False,
                                   convert=True, features=None, **kwargs):

        only_like = self._test_only_like(filter)
        if not only_like:
            new_filter = self._remove_like_from_filter(filter)
            matches_cursor = SddbCursor(
                self,
                new_filter,
                {'_id': 1},
                *args[1:],
                convert=convert,
                features=features,
                **kwargs,
            )
            ids = [x['_id'] for x in matches_cursor]
            similar = self._find_nearest(filter, ids=ids)
        else:
            similar = self._find_nearest(filter)
        if raw:
            return Cursor(self, {'_id': {'$in': similar['ids']}}, **kwargs)
        else:
            return SddbCursor(self, {'_id': {'$in': similar['ids']}}, convert=convert,
                              features=features,
                              scores=dict(zip(similar['ids'], similar['scores'])), **kwargs)

    def find(self, filter=None, *args, similar_first=False, raw=False,
             features=None, download=False, similar_join=None, **kwargs):
        """
        Behaves like MongoDB *find* with exception of "$like" operator.

        :param filter: filter dictionary
        :param *args: args passed to super()
        :param similar_first: toggle to *True* to first find similar things, and then
                              apply filter to these things
        :param raw: toggle to *True* to not convert bytes to Python objects but return raw bytes
        :param features: dictionary of model outputs to replace for dictionary elements
        :param download: toggle to *True* in case query has downloadable "_content" components
        :param similar_join: replace ids by documents
        :param **kwargs: kwargs to be passed to super()
        """

        if download:
            filter = copy.deepcopy(filter)
            assert '_id' not in filter
            filter['_id'] = 0
            urls = self._gather_urls([filter])[0]
            if urls:
                filter = self._download_content(documents=[filter], download_folder='/tmp',
                                                timeout=None, raises=True)[0]
                filter = convert_types(filter, converters=self.converters)
            del filter['_id']

        if filter is None:
            filter = {}
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
                return SddbCursor(self, filter, *args, features=features,
                                  similar_join=similar_join, **kwargs)

    def _delete_objects(self, type, objects=None, force=False):
        if objects is None:
            objects = self[f'_{type}'].distinct('name')
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

    def delete_neighbourhood(self, name):
        info = self['_neighbourhoods'].find_one({'name': name})
        si_info = self['_semantic_indexes'].find_one({'name': info['semantic_index']})
        model = next(m for m in si_info['models'] if self._model_info[m]['active'])
        filter_ = self._model_info[model]['filter']
        n_documents = self.count_documents(filter_)
        if click.confirm(f'Removing neighbourhood "{name}", this will affect {n_documents}'
                         ' documents. Are you sure?', default=False):
            self['_neighbourhoods'].delete_one({'name': name})
            self.update_many(filter_, {'$unset': {'_like': name}}, refresh=False)
        else:
            print('aborting')

    def delete_converters(self, converters=None, force=False):
        return self._delete_objects('converters', objects=converters, force=force)

    def delete_metrics(self, metrics=None, force=False):
        return self._delete_objects('metrics', objects=metrics, force=force)

    def delete_losses(self, losses=None, force=False):
        return self._delete_objects('losses', objects=losses, force=force)

    def delete_measures(self, measures=None, force=False):
        return self._delete_objects('measures', objects=measures, force=force)

    def delete_imputation(self, name, force=False):
        self.delete_model(name, force=force)
        self['_imputations'].delete_many({'name': name})

    def delete_semantic_index(self, name, force=False):
        r = self['_semantic_indexes'].find_one({'name': name}, {'models': 1})
        if r is None:
            return
        models = r['models']
        combined_filter = {'$and': [self._model_info[m].get('filter', {}) for m in models
                                    if self._model_info[m]['active']]}
        if not force:
            n_documents = self.count_documents(combined_filter)
        if force or click.confirm(f'Are you sure you want to delete this semantic index: {name}; '
                                  f'{n_documents} documents will be affected.'):
            for m in models:
                self.delete_model(m, force=True)
            self['_semantic_indexes'].delete_one({'name': name})

    def delete_model(self, name, force=False):
        all_models = self.list_models()
        if '.' in name:
            raise Exception('must delete parent model in order to delete <model>.<attribute>')
        models = [name]
        children = [y for y in all_models if y.startswith(f'{name}.')]
        models.extend(children)
        if not force:
            combined_filter = {'$and': [self._model_info[m].get('filter', {}) for m in models]}
            n_documents = self.count_documents(combined_filter)
        if force or click.confirm(f'Are you sure you want to delete these models: {models}; '
                                  f'{n_documents} documents will be affected.',
                                  default=False):
            deleted_info = [self._model_info[m] for m in models]
            _ = self._delete_objects(
                'models', objects=[x for x in models if '.' not in x], force=True
            )
            for m in models:
                del self._model_info[m]
            for r in deleted_info:
                print(f'unsetting output field _outputs.{r["key"]}.{r["name"]}')
                super().update_many(
                    r.get('filter', {}),
                    {'$unset': {f'_outputs.{r["key"]}.{r["name"]}': 1}}
                )

    def delete_one(
        self,
        filter,
        *args,
        **kwargs,
    ):
        id_ = super().find_one(filter, {'_id': 1})
        if id_ is None:
            return
        super().delete_one({'_id': id_}, *args, **kwargs)

    def delete_many(
        self,
        filter,
        *args,
        **kwargs,
    ):
        ids = [r['_id'] for r in super().find(filter, {'_id': 1})]
        if not ids:
            return
        super().delete_many(filter, *args, **kwargs)

    def list_models(self, training=False):
        if training is None:
            return self['_models'].distinct('name')
        else:
            return self['_models'].distinct('name', {'training': training})

    def list_converters(self):
        return self['_converters'].distinct('name')

    def list_losses(self):
        return self['_losses'].distinct('name')

    def list_measures(self):
        return self['_measures'].distinct('name')

    def list_splitters(self):
        return self['_splitters'].distinct('name')

    def list_metrics(self):
        return self['_metrics'].distinct('name')

    def list_semantic_indexes(self):
        return self['_semantic_indexes'].distinct('name')

    def list_imputations(self):
        return self['_imputations'].distinct('name')

    def _train_imputation(self, name):

        if not self.single_thread:
            return sddb_requests.jobs.process(self.database.name, self.name,
                                              '_train_imputation', name)

        info = self['_imputations'].find_one({'name': name})

        self['_models'].update_one({'name': info['name']}, {'$set': {'training': True}})

        model = self.models[info['model']]
        target = self.models[info['target']]
        loss = self.losses[info['loss']]
        metrics = {k: self.metrics[k] for k in info['metrics']}

        fields = (
            self._model_info[info['model']]['key'],
            self._model_info[info['target']]['key'],
        )

        training.train_tools.ImputationTrainer(
            name,
            cf['mongodb'],
            self.database.name,
            self.name,
            models=(model, target),
            fields=fields,
            save_names=(info['model'], info['target']),
            loss=loss,
            metrics=metrics,
            **info['trainer_kwargs'],
            save=self._replace_model,
            features=info.get('features', {}),
            filter=info['filter'],
            projection=info['projection'],
            n_epochs=info['n_epochs'],
        ).train()

    def _replace_model(self, name, object):
        r = self['_models'].find_one({'name': name})
        assert name in self.list_models(training=None), 'model doesn\'t exist to replace'
        file_id = self._create_pickled_file(object)
        self.filesystem.delete(r['object'])
        self['_models'].update_one({'name': name}, {'$set': {'object': file_id}})

    def create_imputation(self, name, model, loss, target,
                          metrics=None, filter=None, projection=None,
                          n_epochs=20, features=None, **trainer_kwargs):

        if target is not None and isinstance(target, dict):
            self.create_model(**target, active=False)
            target_name = target['name']
        else:
            assert target in self.list_models()
            target_name = target

        if isinstance(model, dict):
            model['process_docs'] = False
            self.create_model(**model)
            model_name = model['name']
        else:
            assert model in self.list_models()
            model_name = model

        if not isinstance(loss, str):
            self.create_loss(**loss)
            loss = loss['name']

        if metrics is not None:
            for i, metric in enumerate(metrics):
                if isinstance(metric, str):
                    continue
                if isinstance(metric, dict):
                    self.create_metric(metric['name'], metric['object'])
                    metrics[i] = metric['name']

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
            'features': features,
        })
        self._train_imputation(name)

        self['_models'].update_one({'name': model_name}, {'$set': {'training': False}})
        r = self['_models'].find_one({'name': model_name})
        ids = [r['_id'] for r in self.find(r.get('filter', {}), {'_id': 1})]

        if self.single_thread:
            self.process_documents_with_model(model_name, ids, **r.get('loader_kwargs', {}),
                                              verbose=True)
        else:
            return sddb_requests.jobs.process(
                self.database.name,
                self.name,
                'process_documents_with_model',
                model_name=model_name,
                ids=ids,
                **r.get('loader_kwargs', {}),
                verbose=False,
            )

    def _train_semantic_index(self, name):

        if self.remote:
            return sddb_requests.jobs.process(self.database.name, self.name,
                                              '_train_semantic_index', name)

        info = self['_semantic_indexes'].find_one({'name': name})
        save_names = info['models']
        features = info.get('features', None)
        models = []
        keys = []
        for save_name in save_names:
            r = self['_models'].find_one({'name': save_name})
            keys.append(r['key'])
            models.append(self.models[save_name])

        metrics = {}
        for metric in info['metrics']:
            metrics[metric] = self.metrics[metric]

        loss = self.losses[info['loss']]
        splitter = info.get('splitter')
        if splitter:
            splitter = self.splitters[info['splitter']]

        training.train_tools.RepresentationTrainer(
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
        ).train()

        self.semantic_index = name
        for model in self.semantic_index['active_models']:
            filter_ = self._model_info[model].get('filter', {})
            if '_fold' in filter_:
                del filter_['_fold']
            self['_models'].update_one({'name': model}, {'$set': {'filter': filter_}})
            ids = [x['_id'] for x in self.find(filter_, {'_id': 1})]
            self.process_documents_with_model(
                model,
                ids,
                verbose=True,
                **self._model_info.get('loader_kwargs', {}),
                max_chunk_size=self._model_info.get('max_chunk_size'),
            )

    def create_semantic_index(self, name, models, measure, metrics=None, loss=None,
                              splitter=None, features=None, **trainer_kwargs):

        for i, man in enumerate(models):
            if isinstance(man, str):
                assert man in self.list_models()
                continue
            self.create_model(**man, process_docs=False)
            models[i] = man['name']

        if metrics is not None:
            for i, man in enumerate(metrics):
                if isinstance(man, str):
                    assert man in self.list_metrics()
                    continue
                self.create_metric(**man)
                metrics[i] = man['name']

        if splitter is not None:
            if isinstance(splitter, str):
                assert splitter in self.list_splitters()
            else:
                self.create_splitter(**splitter)
                splitter = splitter['name']

        if isinstance(measure, str):
            assert measure in self.list_measures()
        else:
            self.create_measure(**measure)
            measure = measure['name']

        if loss is not None:
            if len(models) == 1:
                assert splitter is not None, 'need a splitter for self-supervised ranking...'
            if isinstance(loss, str):
                assert loss in self.list_losses()
            else:
                self.create_loss(**loss)
                loss = loss['name']

        if measure is not None:
            if isinstance(measure, str):
                assert measure in self.list_measures()
            else:
                self.create_measure(**measure)

        self['_semantic_indexes'].insert_one({
            'name': name,
            'models': models,
            'metrics': metrics,
            'loss': loss,
            'measure': measure,
            'features': features,
            'trainer_kwargs': trainer_kwargs,
        })
        if loss is None:
            if 'semantic_index' not in self.meta:
                self.update_meta_data('semantic_index', name)
            return
        else:
            active_model = next(m for m in models if self._model_info[m]['active'])
            filter = self._model_info[active_model].get('filter', {})
            filter['_fold'] = 'valid'
            self['_models'].update_one({'name': active_model}, {'$set': {'filter': filter}})

        self._train_semantic_index(name=name)
        self.semantic_index = name
        if not self.single_thread:
            job_ids = []
        for model in self.semantic_index['active_models']:
            filter_ = self._model_info[model].get('filter', {})
            if '_fold' in filter_:
                del filter_['_fold']
            self['_models'].update_one({'name': model}, {'$set': {'filter': filter_}})
            ids = [x['_id'] for x in self.find(filter_, {'_id': 1})]
            if self.single_thread:
                self.process_documents_with_model(
                    model,
                    ids,
                    verbose=True,
                    **self._model_info.get('loader_kwargs', {}),
                    max_chunk_size=self._model_info.get('max_chunk_size'),
                )
            else:
                job_ids.append(sddb_requests.jobs.process(
                    self.database.name,
                    self.name,
                    'process_documents_with_model',
                    model,
                    ids,
                    verbose=True,
                    **self._model_info.get('loader_kwargs', {}),
                    max_chunk_size=self._model_info.get('max_chunk_size'),
                ))
        return job_ids

    def create_converter(self, name, object):
        """
        Create a converter directly from a Python object.

        :param name: name of converter
        :param object: Python object
        """
        return self._create_object('converters', name, object)

    def create_splitter(self, name, object):
        """
        Create a document splitter directly from a Python object.

        :param name: name of splitter
        :param object: Python object
        """
        return self._create_object('splitters', name, object)

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

    def update_meta_data(self, key, value):
        """
        Update key-value pair of collection meta-data (*Collection.meta*).

        :param key: key
        :param value: value
        """
        self['_meta'].update_one({}, {'$set': {key: value}}, upsert=True)
        self._get_meta()

    def set_meta_data(self, r):
        """
        Replace meta data with new record.

        :param r: record to replace meta-data by.
        """
        r_current = self['_meta'].find_one()
        if r_current is not None:
            self['_meta'].replace_one({'_id': r['_id']}, r, upsert=True)
        else:
            self['_meta'].insert_one(r)
        self._get_meta()

    def replace_one(
        self,
        filter,
        replacement,
        *args,
        refresh=False,
        **kwargs,
    ):
        """
        Replace a document in the database. The outputs of models will be refreshed for this
        document.

        :param filter: MongoDB like filter
        :param *args: args to be passed to super()
        :param refresh: Toggle to *False* to not process document again with models.
        :param **kwargs: kwargs to be passed to super()
        """
        id_ = super().find_one(filter, *args, **kwargs)['_id']
        result = super().replace_one({'_id': id_}, replacement, *args, **kwargs)
        if refresh and self.list_models():
            self._process_documents([id_])
        return result