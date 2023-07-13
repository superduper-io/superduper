from collections import defaultdict
import hashlib
import multiprocessing
import networkx
import random

from pymongo import UpdateOne
from pymongo.collection import Collection as BaseCollection
from pymongo.cursor import Cursor
import torch.utils.data
import tqdm

from sddb import cf
from sddb import models
from sddb.lookup import hashes
from sddb.models.converters import decode, encode
from sddb import requests as sddb_requests
from sddb.training.loading import BasicDataset
from sddb.utils import apply_model, unpack_batch, MongoStyleDict, Downloader


class ArgumentDefaultDict(defaultdict):
    def __getitem__(self, item):
        if item not in self.keys():
            self[item] = self.default_factory(item)
        return super().__getitem__(item)


class SddbCursor(Cursor):
    def __init__(self, collection, *args, features=None, convert=True, **kwargs):
        print(args)
        super().__init__(collection, *args, **kwargs)
        self.attr_collection = collection
        self.features = features
        self.convert = convert

    @staticmethod
    def convert_types(r, convert=True):
        for k in r:
            if isinstance(r[k], dict):
                if '_content' in r[k]:
                    if 'bytes' in r[k]['_content']:
                        if convert:
                            r[k] = decode(r[k]['_content']['converter'], r[k]['_content']['bytes'])
                        else:
                            pass
                    elif 'path' in r[k]['_content']:
                        try:
                            with open(r[k]['_content']['path'], 'rb') as f:
                                if convert:
                                    r[k] = decode(r[k]['_content']['converter'], f.read())
                                else:
                                    r[k]['_content']['bytes'] = f.read()
                        except FileNotFoundError:
                            return
                    else:
                        raise NotImplementedError(
                            f'neither "bytes" nor "path" found in record {r}'
                        )
                else:
                    SddbCursor.convert_types(r[k], convert=convert)
        return r

    def next(self):
        r = super().next()
        if self.features is not None and self.features:
            r = MongoStyleDict(r)
            for k in self.features:
                if k != '_base':
                    r[k] = r['_outputs'][k][self.features[k]]
                else:
                    r = {'_base': r['_outputs'][k][self.features[k]]}
        r = self.convert_types(r, convert=self.convert)
        if r is None:
            return self.next()
        else:
            if '_base' in r:
                return r['_base']
            return r

    __next__ = next


class Collection(BaseCollection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._meta = None
        self._semantic_index = self.meta.get('semantic_index')
        self._semantic_index_data = self['_semantic_indexes'].find_one({
            'name': self._semantic_index
        })
        self._hash_set = None
        self._model_info = ArgumentDefaultDict(self._get_model_info)
        self._models = ArgumentDefaultDict(self._load_model)
        self._all_hash_sets = ArgumentDefaultDict(self._load_hashes)
        self.single_thread = cf.get('single_thread', True)
        self.remote = cf.get('remote', False)
        self.download_timeout = 2

    def _load_model(self, name):
        manifest = self['_models'].find_one({'name': name})
        if manifest is None:
            raise Exception(f'No such model "{name}" has been registered.')
        m = models.loading.load(manifest)
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
    def models(self):
        return self._models

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
        filter = self._model_info[name].get('filter', {})
        key = self._model_info[name].get('key', '_base')
        n_docs = self.count_documents(filter)
        c = self.find(filter, {f'_outputs.{key}.{name}': 1})
        loaded = []
        ids = []
        docs = tqdm.tqdm(c, total=n_docs)
        docs.set_description(f'loading hashes: "{name}"')
        for r in docs:
            h = r['_outputs'][key][name]
            loaded.append(h)
            ids.append(r['_id'])
        return hashes.HashSet(torch.stack(loaded), ids)

    @property
    def hash_set(self):
        if self.semantic_index_name is None:
            raise Exception('No semantic index has been set!')
        active_key = next(m['name'] for m in self.semantic_index['models'] if m['active'])
        return self._all_hash_sets[active_key]

    def process_documents_with_model(self, model_name, ids=None, batch_size=10,
                                     verbose=False):
        if 'requires' not in self._model_info[model_name]:
            filter = {'_id': {'$in': ids}}
        else:
            filter = {'_id': {'$in': ids},
                      '$exists': {self._model_info[model_name]['requires']: 1}}
        documents = list(self.find(
            filter,
            features=self._model_info[model_name].get('features', {})
        ))
        ids = [r['_id'] for r in documents]
        for r in documents:
            del r['_id']
        key = self._model_info[model_name].get('key', '_base')
        if key != '_base':
            passed_docs = [r[key] for r in documents]
        else:
            passed_docs = documents
        model = self._models[model_name]
        inputs = BasicDataset(passed_docs, model.preprocess)
        if hasattr(model, 'forward'):
            loader = torch.utils.data.DataLoader(inputs, batch_size=batch_size)
            if verbose:
                loader = tqdm.tqdm(loader)
                loader.set_description(f'processing with {model_name}')
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

        if 'converter' in self._model_info[model_name]:
            tmp = [
                {model_name: {
                    '_content': {
                        'bytes': encode(self._model_info[model_name]['converter'], x),
                        'converter': self._model_info[model_name]['converter']
                    }
                }}
                for x in outputs
            ]
        else:
            tmp = [{model_name: out} for out in outputs]
        key = self._model_info[model_name].get('key', '_base')
        for i, r in enumerate(tmp):
            if 'target' not in self._model_info[model_name]:
                self.bulk_write([
                    UpdateOne({'_id': id},
                              {'$set': {f'_outputs.{key}.{model_name}': r[model_name]}})
                    for id in ids
                ])
            else:
                self.bulk_write([
                    UpdateOne({'_id': id},
                              {'$set': {
                                  self._model_info[model_name]['target']: r[model_name]
                              }})
                    for id in ids
                ])
        return tmp

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

    def _process_documents(self, ids, batch_size=10, verbose=False, blocking=False):
        if self.single_thread:
            self.download_content(ids=ids)
        else:
            job_ids = defaultdict(lambda: [])
            download_id = sddb_requests.jobs.download_content(
                database=self.database.name,
                collection=self.name,
                ids=ids,
            )
        if not self.active_models:
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
                        model_name=model, ids=sub_ids, batch_size=batch_size, verbose=verbose,
                    )
                    if self._model_info[model].get('download', False):
                        self.download_content(ids=sub_ids)
                else:
                    if iteration == 0:
                        dependencies = [download_id]
                    else:
                        dependencies = sum([
                            job_ids[dep]
                            for dep in self._model_info[model]['dependencies']
                        ], [])
                    process_id = sddb_requests.jobs.process_documents_with_model(
                        database=self.database.name,
                        collection=self.name,
                        model_name=model,
                        ids=ids,
                        batch_size=batch_size,
                        verbose=verbose,
                        blocking=blocking,
                        dependencies=dependencies,
                    )
                    job_ids[model].append(process_id)
                    if self._model_info[model].get('download', False):
                        download_id = sddb_requests.jobs.download_content(
                            database=self.database.name,
                            collection=self.name,
                            ids=sub_ids,
                            dependencies=(process_id,)
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

    def download_content(self, ids):
        assert ids is not None
        documents = list(self.find({'_id': {'$in': ids}}, {'_outputs': 0}, raw=True))
        urls, keys, ids = self._gather_urls(documents)
        if not urls:
            return
        files = []
        for url in urls:
            files.append(
                f'{self.meta["data"]}/{hashlib.sha1(url.encode("utf-8")).hexdigest()}'
            )
        downloader = Downloader(
            urls=urls,
            files=files,
            n_workers=self.meta.get('n_download_workers', 0),
            timeout=self.download_timeout
        )
        downloader.go()
        self.bulk_write([
            UpdateOne({'_id': id_}, {'$set': {f'{key}._content.path': file}})
            for id_, key, file in zip(ids, keys, files)
        ])

    def insert_many(
        self,
        documents,
        *args,
        verbose=False,
        **kwargs,
    ):
        if 'valid_probability' in self.meta:
            for document in documents:
                r = random.random()
                document['_fold'] = 'valid' if r < self.meta['valid_probability'] else 'train'
        output = super().insert_many(documents, *args, **kwargs)
        self._process_documents(output.inserted_ids, verbose=verbose)
        return output

    def update_one(
        self,
        filter,
        refresh=True,
        *args,
        **kwargs,
    ):
        if refresh and self.list_models():
            id_ = super().find_one(filter, *args, **kwargs)['_id']
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

    def find_nearest(self, semantic_index, h=None, n=100, model=None, document=None,
                     ids=None):
        if self.single_thread:
            if model is not None:
                model = self._models[model]
                with torch.no_grad():
                    h = apply_model(model, document, True)[0]
            si = self.semantic_index['name']
            self.semantic_index = semantic_index
            if ids is None:
                hash_set = self.hash_set
            else:
                hash_set = self.hash_set[ids]
            output = hash_set.find_nearest_from_hash(h, n=n)
            self.semantic_index = si
            return output
        else:
            return sddb_requests.hash_set.find_nearest(
                self.database.name,
                self.name,
                self.semantic_index['name'],
                document=document,
                model=model,
            )

    def find_one(self, filter=None, *args, similar_first=True, raw=False, features=None,
                 convert=True, **kwargs):
        if self.remote:
            return sddb_requests.client.find_one(
                self.database.name,
                self.name,
                filter,
                *args,
                similar_first=similar_first,
                raw=raw,
                features=features,
                **kwargs,
            )
        cursor = self.find(filter, *args,
                           raw=raw, features=features, convert=convert, **kwargs)
        for result in cursor.limit(-1):
            return result
        return None

    def _find_similar(self, filter, like_place, ids=None):
        document = (
            filter['$like']['document'] if like_place == '_base'
            else filter[like_place]['$like']['document']
        )
        model = next(
            man['name'] for man in self.semantic_index['models']
            if man.get('key', '_base') == like_place
        )
        return self.find_nearest(
            semantic_index=self.semantic_index['name'],
            model=model,
            document=document,
            n=filter[like_place]['$like']['n'] if like_place != '_base' else filter['$like']['n'],
            ids=ids,
        )

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
        for k in r:
            if isinstance(r[k], dict):
                r[k] = Collection._remove_like_from_filter(r[k])
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

    def _find_similar_then_matches(self, filter, like_place, *args, raw=False,
                                   convert=True, **kwargs):
        similar = self._find_similar(filter, like_place)
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
            return SddbCursor(self, filter, *args, convert=convert, **kwargs)

    def _find_matches_then_similar(self, filter, like_place, *args, raw=False,
                                   convert=True, **kwargs):
        only_like = self._test_only_like(filter)
        if not only_like:
            new_filter = self._remove_like_from_filter(filter)
            matches_cursor = SddbCursor(
                self,
                new_filter,
                {'_id': 1},
                *args,
                convert=convert,
                **kwargs,
            )
            ids = [x['_id'] for x in matches_cursor]
            similar = self._find_similar(filter, like_place, ids=ids)
        else:
            similar = self._find_similar(filter, like_place)
        if raw:
            return Cursor(self, {'_id': {'$in': similar['ids']}})
        else:
            return SddbCursor(self, {'_id': {'$in': similar['ids']}}, convert=convert)

    def find(self, filter=None, *args, similar_first=True, raw=False,
             features=None, convert=True, **kwargs):
        if filter is None:
            filter = {}
        like_place = self._find_like_operator(filter)
        if like_place is not None:
            filter = MongoStyleDict(filter)
            if similar_first:
                return self._find_similar_then_matches(filter, like_place, *args, raw=raw,
                                                       convert=convert, **kwargs)
            else:
                return self._find_matches_then_similar(filter, like_place, *args, raw=raw,
                                                       convert=convert, **kwargs)
        else:
            if features is not None:
                assert not raw, 'only use features with SddbCursor'
                kwargs['features'] = features
            if raw:
                return Cursor(self, filter, *args, **kwargs)
            else:
                return SddbCursor(self, filter, *args, convert=convert, **kwargs)

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

    def list_models(self):
        return self['_models'].distinct('name')

    def list_semantic_indexes(self):
        return self['_semantic_indexes'].distinct('name')

    def list_imputations(self):
        raise NotImplementedError

    def create_imputation(self, models):
        '''
        manifest = {
            'name': '<thename>',
            'input': ['<key1>'],
            'label': ['<key2>'],
            'models': {
                '<key1>': '<existing-model>',
                '<key2[optional]>': {
                    'name': '<name-of-model-for-label-encoding>',
                    'type': ...,
                    'active': False # one should be inactive,
                },
            },
            'metrics': [
                {
                    'name': 'p@1',
                    'type': ...,
                    'args': {...},
                }
            ],
            'filter': '<active-set-of-model>',
        }
        '''

    def create_loss(self, manifest):
        '''
        manifest = {
            'name': '<thename>',
            'type': 'import',
            'args': {
                'path': '...',
                'kwargs': {...}
            },
        }
        '''
        assert manifest['name'] not in self['_losses'].distinct('name')
        self['_losses'].insert_one(manifest)

    def compile_neighbours(self):
        '''
        Find ids of similar items, and add these to the records (used for retrieval
        enhanced deep learning)

        Useful for e.g. retrieval enhanced ML
        '''
        # TODO do this
        ...

    def create_semantic_index(self, manifest):
        '''
        manifest = {
            'name': '<thename>',
            'models': [
                '<existing-model-name>',
                {
                    'name': '<name-of-model-to-be-added>',
                    'type': ...
                    'active': True/False # one should be active
                    'filter': '<active-set-of-model>',
                    'key': 'the-key'
                },
            ],
            'metrics': [
                {
                    'name': 'p@1',
                    'type': ...,
                    'args': {...},
                }
            ],
            'loss': 'loss to use[optional]',
            'measure': 'dot',
        }
        '''
        for i, man in enumerate(manifest['models']):
            if isinstance(man, str):
                continue
            self.create_model(man)
            manifest['models'][i] = man['name']
        self['_semantic_indexes'].insert_one(manifest)
        if 'semantic_index' not in self.meta:
            self.update_meta_data('semantic_index', manifest['name'])

    def create_model(self, manifest):
        '''
        manifest = {
            'name': '<model-name>',
            'type': '<import-type>',
            'args': '<arguments-to-import-type>',
            'filter': '<active-set-of-model>',
            'converter': '<import-path-of-converter[optional]>',
            'object': '<python-object[optional]>',
            'active': '<toggle-to-false-for-not-watching-inserts[optional]',
        }
        '''
        assert manifest['name'] not in self['_models'].distinct('name'), \
            f'Model {manifest["name"]} already exists!'
        if manifest['type'] == 'in_memory':
            self._models[manifest['name']] = manifest['object']
        self['_models'].insert_one({k: v for k, v in manifest.items() if k != 'object'})

    def create_metric(self, manifest):
        '''
        manifest = {
            'name': '<metric-name>',
            'type': '<object-type>',  # import, pickle, ...
            'args': '<arguments-to-import-type>',
            'task': '<imputation/classification>',
        }
        '''
        assert manifest['name'] not in self['_metrics'].distinct('name'), \
            f'Metric {manifest["name"]} already exists!'
        self['_metrics'].insert_one(manifest)

    def load_metric(self, name):
        r = self['_metrics'].find_one({'name': name})
        m = models.loading.load(r)
        return m

    def update_meta_data(self, key, value):
        self['_meta'].update_one({}, {'$set': {key: value}}, upsert=True)
        self._get_meta()

    def set_meta_data(self, r):
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
        **kwargs,
    ):
        if self.list_models():
            id_ = super().find_one(filter, *args, **kwargs)['_id']
        result= super().replace_one(filter, replacement, *args, **kwargs)
        if self.list_models():
            self._process_documents([id_])
        return result