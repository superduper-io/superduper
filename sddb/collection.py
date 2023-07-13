import hashlib
import random
from collections import defaultdict

from pymongo import UpdateOne
from pymongo.collection import Collection as BaseCollection
from pymongo.cursor import Cursor
from sddb import models
from sddb.lookup import hashes
import torch.utils.data
import tqdm

from sddb.models.converters import decode, encode
from sddb.training.loading import BasicDataset
from sddb.utils import apply_model, unpack_batch, MongoStyleDict, Downloader
from sddb import requests as sddb_requests


class ArgumentDefaultDict(defaultdict):
    def __getitem__(self, item):
        if item not in self.keys():
            self[item] = self.default_factory(item)
        return super().__getitem__(item)


class SddbCursor(Cursor):
    def __init__(self, collection, *args, **kwargs):
        super().__init__(collection, *args, **kwargs)
        self.attr_collection = collection

    @staticmethod
    def _convert_types(r):
        for k in r:
            if isinstance(r[k], dict):
                if '_content' in r[k]:
                    if 'bytes' in r[k]['_content']:
                        r[k] = decode(r[k]['_content']['converter'], r[k]['_content']['bytes'])
                    elif 'path' in r[k]['_content']:
                        with open(r[k]['_content']['path'], 'rb') as f:
                            r[k] = decode(r[k]['_content']['converter'], f.read())
                    else:
                        raise NotImplementedError
                else:
                    SddbCursor._convert_types(r[k])

    def next(self):
        r = super().next()
        self._convert_types(r)
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
        self.single_thread = True

    def _load_model(self, name):
        manifest = self['_models'].find_one({'name': name})
        if manifest is None:
            raise Exception(f'No such model "{name}" has been registered.')
        m = models.load(manifest)
        m.eval()
        return m

    def _get_meta(self):
        self._meta = self['_meta'].find_one()

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
        if self._meta is None:
            self._get_meta()
        return self._meta if self._meta is not None else {}

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
        return self._all_hash_sets[self.semantic_index['models'][self.semantic_index['target']]]

    def process_documents_with_model(self, model_name, ids=None, documents=None, batch_size=10,
                                     verbose=False):
        if documents is None:
            assert ids is not None
        elif ids is None:
            assert documents is not None
        if documents is None:
            documents = list(self.find({'_id': {'$in': ids}}, {'_outputs': 0}))
        model = self._models[model_name]
        inputs = BasicDataset(documents, model.preprocess)
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
        key = self._model_info[model].get('key', '_base')
        for i, r in enumerate(tmp):
            if '_outputs' not in documents[i]:
                documents[i]['_outputs'] = {}
            if key not in documents[i]['_outputs']:
                documents[i]['_outputs'][key] = {}
            documents[i]['_outputs'][key].update(r)
        self.bulk_write([
            UpdateOne({'_id': id},
                      {'$set': {f'_outputs.{key}.{model}': r['_outputs'][key][model]}})
            for id, r in zip(ids, documents)
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

    def _process_documents(self, ids, documents=None, batch_size=10, verbose=False):
        # TODO have concept of precedence (some computations depending on others)
        if documents is not None:
            documents = [MongoStyleDict(r) for r in documents]
        filters = []
        for model in self.active_models:
            filters.append(self._model_info[model].get('filter', {}))
        filter_lookup = {self.dict_to_str(f): f for f in filters}
        lookup = {}
        for filter_str in filter_lookup:
            if filter_str not in lookup:
                ids = [
                    r['_id']
                    for r in super().find({
                        '$and': [{'_id': {'$in': ids}}, filter_lookup[filter_str]]
                    })
                ]
                if documents is not None:
                    lookup[filter_str] = {
                        'ids': ids,
                        'documents': [d for id_, d in zip(ids, documents) if id_ in ids]
                    }
                else:
                    lookup[filter_str] = {'ids': ids}

        for model in self.active_models:
            filter = self._model_info[model].get('filter', {})
            key = self._model_info[model].get('key', '_base')
            filter_str = self.dict_to_str(filter)
            sub_ids = lookup[filter_str]['ids']
            if not sub_ids:
                continue
            if documents is not None:
                sub_documents = lookup[filter_str]['documents']
                if key != '_base':
                    passed_docs = [r[key] for r in sub_documents]
                else:
                    passed_docs = sub_documents
            else:
                passed_docs = None

            if self.single_thread:
                self.process_documents_with_model(
                    model_name=model, ids=ids, documents=passed_docs, batch_size=batch_size,
                    verbose=verbose,
                )
            else:
                sddb_requests.jobs.process_documents_with_model(
                    database=self.database.name,
                    collection=self.name,
                    model_name=model,
                    ids=ids,
                    batch_size=batch_size,
                    verbose=verbose,
                )

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
                                    [document] if self.single_thread else None)
        return output

    def _download_content(self, documents):
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
        self._download_content(documents)
        if self.list_models():
            self._process_documents(output.inserted_ids, documents if self.single_thread else None,
                                    verbose=verbose)
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
            self._process_documents([document['_id']], [document] if self.single_thread else None)
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
            documents = super().find({'_id': {'$in': ids}}, {'_outputs': 0})
            self._process_documents(ids, documents=documents if self.single_thread else None)
        return result

    def find_nearest_from_hash(self, semantic_index, h, n):
        if self.single_thread:
            si = self.semantic_index['name']
            self.semantic_index = semantic_index
            output = self.hash_set.find_nearest_from_hash(h, n=n)
            self.semantic_index = si
            return output
        else:
            return sddb_requests.hash_set.find_nearest_from_hash(
                self.database.name,
                self.name,
                self.semantic_index['name'],
                h,
                n
            )

    def find_one(self, filter=None, *args, raw=False, **kwargs):
        if filter is None:
            filter = {}
        like_place = self._find_like_operator(filter)
        if like_place is not None:
            assert self.semantic_index is not None, 'semantic index cannot be None for $like'
            model = self.models[self.semantic_index['models'][like_place]]
            if like_place != '_base':
                document = MongoStyleDict(filter)[like_place]['$like']['document']
                n = MongoStyleDict(filter)[like_place]['$like']['n']
            else:
                document = filter['$like']['document']
                n = filter['$like']['n']
            output = apply_model(model, document, True)[0]
            # similar = self.hash_set.find_nearest_from_hash(output, n=n)
            similar = self.find_nearest_from_hash(self.semantic_index['name'], output, n)
            only_like = self._test_only_like(filter)
            if not only_like:
                new_filter = self._remove_like_from_filter(filter)
                new_filter = {'$and': [
                    new_filter,
                    {'_id': {'$in': similar['ids']}}
                ]}
            else:
                new_filter = {'_id': similar['ids'][0]}
        else:
            new_filter = filter

        cursor = self.find(new_filter, *args, raw=raw, **kwargs)
        for result in cursor.limit(-1):
            return result
        return None

    def _find_similar(self, filter, like_place, ids=None):
        if ids is None:
            hash_set = self.hash_set
        else:
            hash_set = self.hash_set[ids]
        model = self.models[self.semantic_index['models'][like_place]]
        if like_place == '_base':
            output = apply_model(model, filter['$like']['document'], True)[0]
            similar = hash_set.find_nearest_from_hash(output, n=filter['$like']['n'])
        else:
            output = apply_model(model, filter[like_place]['$like']['document'], True)[0]
            similar = hash_set.find_nearest_from_hash(output, n=filter[like_place]['$like']['n'])
        return similar

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

    def _find_similar_then_matches(self, filter, like_place, *args, raw=False, **kwargs):
        similar = self._find_similar(filter, like_place)
        only_like = self._test_only_like(filter)
        if not only_like:
            new_filter = self._remove_like_from_filter(filter)
            return (SddbCursor if not raw else Cursor)(self,
                {
                    '$and': [
                        new_filter,
                        {'_id': {'$in': similar['ids']}}
                    ]
                },
                *args,
                **kwargs,
            )
        else:
            return (SddbCursor if not raw else Cursor)(
                self,
                {'_id': {'$in': similar['ids']}},
                *args,
                **kwargs,
            )

    def _find_matches_then_similar(self, filter, like_place, *args, raw=False, **kwargs):
        only_like = self._test_only_like(filter)
        if not only_like:
            new_filter = self._remove_like_from_filter(filter)
            matches_cursor = SddbCursor(
                self,
                new_filter,
                {'_id': 1},
                *args,
                **kwargs,
            )
            ids = [x['_id'] for x in matches_cursor]
            similar = self._find_similar(filter, like_place, ids=ids)
        else:
            similar = self._find_similar(filter, like_place)
        return (SddbCursor if not raw else Cursor)(
            self,
            {'_id': {'$in': similar['ids']}},
            *args,
            **kwargs,
        )

    def find(self, filter=None, *args, similar_first=True, raw=False, **kwargs):
        if filter is None:
            filter = {}

        like_place = self._find_like_operator(filter)
        if like_place is not None:
            filter = MongoStyleDict(filter)
            if similar_first:
                return self._find_similar_then_matches(filter, like_place, *args, raw=raw, **kwargs)
            else:
                return self._find_matches_then_similar(filter, like_place, *args, raw=raw, **kwargs)
        else:
            return (SddbCursor if not raw else Cursor)(self, filter, *args, **kwargs)

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
        '''
        # TODO do this
        ...

    def create_semantic_index(self, manifest):
        '''
        manifest = {
            'name': '<thename>',
            'keys': ['<key1>', '<key2>'],
            'models': {
                '<key1>': '<existing-model>',
                '<key2[optional]>': {
                    'name': '<name-of-model-to-be-added>',
                    'type': ...
                    'active': True/False # one should be active
                    'filter': '<active-set-of-model>',
                },
            },
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
        for key in manifest['models']:
            if isinstance(manifest['models'][key], str):
                continue
            self.create_model(manifest['models'][key])
            manifest['models'][key] = manifest['models'][key]['name']
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
            'converter': '<import-path-of-converter[optional]>'
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
            'task': 'imputation/classification',
        }
        '''
        assert manifest['name'] not in self['_metrics'].distinct('name'), \
            f'Metric {manifest["name"]} already exists!'
        self['_metrics'].insert_one(manifest)

    def load_metric(self, name):
        r = self['_metrics'].find_one({'name': name})
        m = models.load(r)
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
            self._process_documents([id_], [replacement] if self.single_thread else None)
        return result