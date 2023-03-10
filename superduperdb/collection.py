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
        self._semantic_index = None

    def __getitem__(self, item):
        if item == '_validation_sets':
            return self.database[f'{self.name}._validation_sets']
        return super().__getitem__(item)

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

    def apply_model(self, model, input_, **kwargs):
        if self.remote:
            return superduper_requests.client.apply_model(self.database.name,
                                                          self.name,
                                                          model, input_, **kwargs)
        if isinstance(model, str):
            model = self.models[model]
        return apply_model(model, input_, **kwargs)

    def create_watcher(self, *args, **kwargs):
        return self.database.create_watcher(self.name, *args, **kwargs)

    def create_neighbourhood(self, *args, **kwargs):
        return self.database.create_neighbourhood(self.name, *args, **kwargs)

    def create_validation_set(self, *args, **kwargs):
        return self.database.create_validation_set(self.name, *args, **kwargs)

    def delete_imputation(self, name, force=False):
        """
        Delete imputation from collection

        :param name: Name of imputation
        :param force: Toggle to ``True`` to skip confirmation
        """
        self.database.delete_imputation(self.name, name, force=force)

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
            self.delete_watcher(watcher['model'], watcher['key'], force=force)
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
        replacement = self.convert_types(replacement)
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
        args[0] = self.convert_types(args[0])
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
        if self.remote:
            return superduper_requests.client.unset_hash_set(self.database.name, self.name)
        if self.semantic_index in self._all_hash_sets:
            del self._all_hash_sets[self.semantic_index]

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

    @staticmethod
    def _dict_to_str(d):
        sd = Collection._standardize_dict(d)
        return str(sd)

    def _download_content(self, ids=None, documents=None, timeout=None, raises=True,
                          n_download_workers=None, headers=None):
        import sys
        sys.path.insert(0, os.getcwd())

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
            like = self.convert_types(like)
            return superduper_requests.client.find_nearest(self.database.name, self.name, like,
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
            self.convert_types(r)
        return documents

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
