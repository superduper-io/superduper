from collections import defaultdict
from dataclasses import asdict
import io
import math
import pickle
import random
import time
from typing import Any, Union

import click
import networkx
from bson import ObjectId

from superduperdb import cf, misc
from superduperdb.cluster.client_decorators import model_server, vector_search
from superduperdb.cluster.annotations import Convertible, Tuple, List
from superduperdb.cluster.job_submission import work
from superduperdb.cluster.task_workflow import TaskWorkflow
from superduperdb.core.learning_task import LearningTask
from superduperdb.datalayer.base.query import Insert, Select, Delete, Update
from superduperdb.fetchers.downloads import gather_uris
from superduperdb.misc.special_dicts import ArgumentDefaultDict, MongoStyleDict
from superduperdb.fetchers.downloads import Downloader
from superduperdb.misc import progress
from superduperdb.misc.logger import logging


def validate_representations(database, vs, identifier, metrics):
    raise NotImplementedError


class BaseDatabase:
    """
    Base database connector for SuperDuperDB - all database types should subclass this
    type.
    """

    select_cls = Select
    variety_to_cache_mapping = {'model': 'models', 'metric': 'metrics', 'type': 'types'}

    def __init__(self):
        self.metrics = ArgumentDefaultDict(lambda x: self.load_component(x, 'metric'))
        self.models = ArgumentDefaultDict(lambda x: self.load_component(x, 'model'))
        self.types = ArgumentDefaultDict(lambda x: self.load_component(x, 'type'))

        self.remote = cf.get('remote', False)
        self._type_lookup = None

        self._hash_set = None
        self._all_hash_sets = {}

    def _reload_type_lookup(self):
        self._type_lookup = {}
        for t in self.list_types():
            try:
                for s in self.types[t].types:
                    self._type_lookup[s] = t
            except AttributeError:
                continue

    @property
    def filesystem(self):
        raise NotImplementedError

    @property
    def type_lookup(self):
        if self._type_lookup is None:
            self._reload_type_lookup()
        return self._type_lookup

    def _add_split_to_row(self, r, other):
        raise NotImplementedError

    def _base_insert(self, insert: Insert):
        raise NotImplementedError

    @model_server
    def predict_one(self, model, input_: Convertible(), **kwargs) -> Convertible():
        if isinstance(model, str):
            model = self.models[model]
        return model.predict_one(
            input_, **{k: v for k, v in kwargs.items() if k != 'remote'}
        )

    @model_server
    def predict(self, model, input_: Convertible(), **kwargs) -> Convertible():
        """
        Apply model to input.

        :param model: model or ``str`` referring to an uploaded model
            (see ``self.list_models``)
        :param input_: input_ to be passed to the model.
                       Must be possible to encode with registered types
                       (``self.list_types``)
        :param kwargs: key-values (see ``superduperdb.models.utils.predict``)
        """
        if isinstance(model, str):
            model = self.models[model]
        return model.predict(input_, **kwargs)

    @work
    def apply_watcher(
        self,
        identifier,
        ids=None,
        verbose=False,
        max_chunk_size=5000,
        model=None,
        recompute=False,
        watcher_info=None,
        **kwargs,
    ):
        select = self.select_cls(**watcher_info['select'])

        if watcher_info is None:
            watcher_info = self.get_object_info(identifier, 'watcher')
        if ids is None:
            ids = self._get_ids_from_select(select.select_only_id)
        if max_chunk_size is not None:
            for it, i in enumerate(range(0, len(ids), max_chunk_size)):
                logging.info(
                    'computing chunk '
                    f'({it + 1}/{math.ceil(len(ids) / max_chunk_size)})'
                )
                self.apply_watcher(
                    identifier,
                    ids=ids[i : i + max_chunk_size],
                    verbose=verbose,
                    max_chunk_size=None,
                    model=model,
                    recompute=recompute,
                    watcher_info=watcher_info,
                    remote=False,
                    **kwargs,
                )
            return

        model_info = self.get_object_info(watcher_info['model'], 'model')
        outputs = self._compute_model_outputs(
            model_info,
            ids,
            select,
            key=watcher_info['key'],
            features=watcher_info.get('features', {}),
            model=model,
            predict_kwargs=watcher_info.get('predict_kwargs', {}),
        )
        type_ = model_info.get('type')
        if type_ is not None:
            type_ = self.types[type_]
            outputs = [
                {'_content': {'bytes': type_.encode(x), 'type': model_info['type']}}
                for x in outputs
            ]

        self._write_watcher_outputs(watcher_info, outputs, ids)
        return outputs

    def _build_task_workflow(
        self, select: Select, ids=None, dependencies=(), verbose=True
    ):
        job_ids = defaultdict(lambda: [])
        job_ids.update(dependencies)
        G = TaskWorkflow(self)
        if ids is None:
            ids = self._get_ids_from_select(select.select_only_id)

        G.add_node(
            f'{self.download_content.__name__}()',
            data={
                'task': self.download_content,
                'args': [
                    select,
                ],
                'kwargs': {
                    'ids': ids,
                },
            },
        )
        if not self.list_watchers():
            return G

        for identifier in self.list_watchers():
            G.add_node(
                f'{self.apply_watcher.__name__}({identifier})',
                data={
                    'task': self.apply_watcher,
                    'args': [identifier],
                    'kwargs': {
                        'ids': ids,
                        'verbose': verbose,
                    },
                },
            )

        for identifier in self.list_neighbourhoods():
            G.add_node(
                f'{self.compute_neighbourhood.__name__}({identifier})',
                data={
                    'task': self.compute_neighbourhood,
                    'args': [identifier],
                    'kwargs': {},
                },
            )

        for identifier in self.list_watchers():
            G.add_edge(
                f'{self.download_content.__name__}()',
                f'{self.apply_watcher.__name__}({identifier})',
            )
            deps = self._get_dependencies_for_watcher(identifier)
            for dep in deps:
                G.add_edge(
                    f'{self.apply_watcher.__name__}({dep})',
                    f'{self.apply_watcher.__name__}({identifier})',
                )
                G.add_edge(
                    f'{self.download_content.__name__}()',
                    f'{self.apply_watcher.__name__}({identifier})',
                )

        for identifier in self.list_neighbourhoods():
            watcher_identifier = self._get_watcher_for_neighbourhood(identifier)
            G.add_edge(('watcher', watcher_identifier), ('neighbourhood', identifier))
            G.add_edge(
                f'{self.apply_watcher.__name__}({watcher_identifier})',
                f'{self.compute_neighbourhood.__name__}({identifier})',
            )
        return G

    def cancel_job(self, job_id):
        raise NotImplementedError

    def _compute_model_outputs(
        self,
        model_info,
        _ids,
        select: Select,
        key='_base',
        features=None,
        model=None,
        predict_kwargs=None,
    ):
        logging.info('finding documents under filter')
        features = features or {}
        model_identifier = model_info['identifier']
        if features is None:
            features = {}  # pragma: no cover
        documents = list(self.select(select.select_using_ids(_ids), features=features))
        logging.info('done.')
        if key != '_base' or '_base' in features:
            passed_docs = [r[key] for r in documents]
        else:  # pragma: no cover
            passed_docs = documents
        if model is None:
            model = self.models[model_identifier]
        return model.predict(passed_docs, **(predict_kwargs or {}))

    @work
    def compute_neighbourhood(self, identifier):
        # TODO testing, refactoring...
        info = self.get_object_info(identifier, 'neighbourhood')
        watcher_info = self.get_object_info(identifier, 'watcher')
        select = self.select_cls(**watcher_info['query'])
        ids = self._get_ids_from_select(select.select_only_id)
        logging.info('getting hash set')
        self._load_hashes(
            identifier, measure=info['measure'], hash_set_cls=info['hash_set_cls']
        )
        h = self._all_hash_sets[identifier]
        h = h[ids]
        logging.info(
            f'computing neighbours based on neighbourhood "{identifier}" and '
            f'index "{info["semantic_index"]}"'
        )

        for i in progress.progressbar(range(0, len(ids), info['batch_size'])):
            sub = ids[i : i + info['batch_size']]
            results = h.find_nearest_from_ids(sub, n=info['n'])
            similar_ids = [res['_ids'] for res in results]
            self._update_neighbourhood(sub, similar_ids, identifier, select)

    def convert_from_bytes_to_types(self, r):
        """
        Convert the bson byte objects in a nested dictionary into python objects.

        :param r: dictionary potentially containing non-Bsonable content
        """
        return misc.serialization.convert_from_bytes_to_types(r, self.types)

    def convert_from_types_to_bytes(self, r):
        """
        Convert the non-Bsonable python objects in a nested dictionary into ``bytes``

        :param r: dictionary potentially containing non-Bsonable content
        """
        return misc.serialization.convert_from_types_to_bytes(
            r, self.types, self.type_lookup
        )

    def _create_job_record(self, *args, **kwargs):
        raise NotImplementedError

    def create_learning_task(
        self,
        models,
        keys,
        select,
        identifier=None,
        configuration=None,
        verbose=False,
        validation_sets=(),
        metrics=(),
        keys_to_watch=(),
        features=None,
        serializer='pickle',
    ):
        if identifier is None:
            identifier = '+'.join([f'{m}/{k}' for m, k in zip(models, keys)])

        assert identifier not in self.list_learning_tasks()

        for k in keys_to_watch:
            assert f'{identifier}/{k}' not in self.list_watchers()

        file_id = self._create_serialized_file(configuration, serializer=serializer)

        self._create_component_entry(
            {
                'variety': 'learning_task',
                'identifier': identifier,
                'select': asdict(select),
                'configuration': {'_file_id': file_id},
                'models': models,
                'keys': keys,
                'metrics': metrics,
                'features': features,
                'validation_sets': validation_sets,
                'keys_to_watch': keys_to_watch,
            }
        )
        model_lookup = dict(zip(keys_to_watch, models))
        jobs = []
        if callable(configuration):
            jobs.append(self.fit(identifier))
        for k in keys_to_watch:
            jobs.append(
                self.create_watcher(
                    f'[{identifier}]:{model_lookup[k]}/{k}',
                    model_lookup[k],
                    select,
                    key=k,
                    features=features,
                    predict_kwargs=configuration.get('loader_kwargs', {})
                    if configuration is not None
                    else {},
                    dependencies=[jobs[0]] if jobs else (),
                    verbose=verbose,
                )
            )
        return jobs

    def _create_component_entry(self, info):
        raise NotImplementedError

    def create_component(
        self, object, serializer='pickle', serializer_kwargs=None
    ):
        serializer_kwargs = serializer_kwargs or {}
        assert object.identifier not in self.list_components(object.variety)
        file_id = self._create_serialized_file(
            object, serializer=serializer, serializer_kwargs=serializer_kwargs
        )
        self._create_component_entry(
            {**object.asdict(), 'object': file_id, 'variety': object.variety}
        )
        return object.schedule_jobs(self)

    def _create_plan(self):
        G = networkx.DiGraph()
        for identifier in self.list_watchers():
            G.add_node(('watcher', identifier))
        for identifier in self.list_watchers():
            deps = self._get_dependencies_for_watcher(identifier)
            for dep in deps:
                G.add_edge(('watcher', dep), ('watcher', identifier))
        for identifier in self.list_neighbourhoods():
            watcher_identifier = self._get_watcher_for_neighbourhood(identifier)
            G.add_edge(('watcher', watcher_identifier), ('neighbourhood', identifier))
        assert networkx.is_directed_acyclic_graph(G)
        return G

    def _create_serialized_file(
        self, object, serializer='pickle', serializer_kwargs=None
    ):
        serializer_kwargs = serializer_kwargs or {}
        if serializer == 'pickle':
            with io.BytesIO() as f:
                pickle.dump(object, f, **serializer_kwargs)
                bytes_ = f.getvalue()
        elif serializer == 'dill':
            import dill

            if not serializer_kwargs:
                serializer_kwargs['recurse'] = True
            with io.BytesIO() as f:
                dill.dump(object, f, **serializer_kwargs)
                bytes_ = f.getvalue()
        else:
            raise NotImplementedError
        return self._save_blob_of_bytes(bytes_)

    def create_validation_set(self, identifier, select: Select, chunk_size=1000):
        if identifier in self.list_validation_sets():
            raise Exception(f'validation set {identifier} already exists!')

        data = self.select(select)
        it = 0
        tmp = []
        for r in progress.progressbar(data):
            tmp.append(r)
            it += 1
            if it % chunk_size == 0:
                self._insert_validation_data(tmp, identifier)
                tmp = []
        if tmp:
            self._insert_validation_data(tmp, identifier)

    def delete(self, delete: Delete):
        return self._base_delete(delete)

    def delete_component(self, identifier, variety, force=False):
        info = self.get_object_info(identifier, variety)
        if not info:
            if not force:
                raise Exception(f'"{identifier}": {variety} does not exist...')
            return
        if force or click.confirm(
            f'You are about to delete {variety}: {identifier}, are you sure?',
            default=False,
        ):
            if variety in self.variety_to_cache_mapping:
                try:
                    del getattr(self, self.variety_to_cache_mapping[variety])[identifier]
                except KeyError:
                    pass
            self.filesystem.delete(info['object'])
            self._delete_component_info(identifier, variety)

    def _delete_component_info(self, identifier, variety):
        raise NotImplementedError

    @work
    def download_content(
        self,
        query: Union[Select, Insert],
        ids=None,
        documents=None,
        timeout=None,
        raises=True,
        n_download_workers=None,
        headers=None,
        **kwargs,
    ):
        logging.debug(query)
        logging.debug(ids)
        update_db = False

        if documents is not None:
            pass
        elif isinstance(query, Select):
            update_db = True
            if ids is None:
                documents = list(self.select(query))
            else:
                documents = list(self.select(query.select_using_ids(ids), raw=True))
        else:
            documents = query.documents

        uris, keys, place_ids = gather_uris(documents)
        logging.info(f'found {len(uris)} uris')
        if not uris:
            return

        if n_download_workers is None:
            try:
                n_download_workers = self.get_meta_data(key='n_download_workers')
            except TypeError:
                n_download_workers = 0

        if headers is None:
            try:
                headers = self.get_meta_data(key='headers')
            except TypeError:
                headers = 0

        if timeout is None:
            try:
                timeout = self.get_meta_data(key='download_timeout')
            except TypeError:
                timeout = None

        def update_one(id, key, bytes):
            return self.update(self._download_update(query.table, id, key, bytes))

        downloader = Downloader(
            uris=uris,
            ids=place_ids,
            keys=keys,
            update_one=update_one,
            n_workers=n_download_workers,
            timeout=timeout,
            headers=headers,
            raises=raises,
        )
        downloader.go()
        if update_db:
            return
        for id_, key in zip(place_ids, keys):
            documents[id_] = self._set_content_bytes(
                documents[id_], key, downloader.results[id_]
            )
        return documents

    def _download_update(self, table, id, key, bytes):
        raise NotImplementedError

    def _get_content_for_filter(self, filter):
        if '_id' not in filter:
            filter['_id'] = 0
        uris = gather_uris([filter])[0]
        if uris:
            filter = self.download_content(
                self.name, documents=[filter], timeout=None, raises=True
            )[0]
            filter = self.convert_from_bytes_to_types(filter)
        return filter

    def _get_cursor(self, select: Select, features=None, scores=None):
        raise NotImplementedError

    def _get_dependencies_for_watcher(self, identifier):
        info = self.get_object_info(identifier, 'watcher')
        if info is None:
            return []
        watcher_features = info.get('features', {})
        dependencies = []
        if watcher_features:
            for key, model in watcher_features.items():
                dependencies.append(f'{model}/{key}')
        return dependencies

    def _get_file_content(self, r):
        for k in r:
            if isinstance(r[k], dict):
                if '_file_id' in r[k]:
                    r[k] = self._load_serialized_file(r[k]['_file_id'])
                else:
                    r[k] = self._get_file_content(r[k])
        return r

    def _get_output_from_document(self, r, key, model):
        raise NotImplementedError

    def _get_job_info(self, identifier):
        raise NotImplementedError

    def _get_ids_from_select(self, select: Select):
        raise NotImplementedError

    def _get_raw_cursor(self, select: Select):
        raise NotImplementedError

    def get_meta_data(self, **kwargs):
        raise NotImplementedError

    def _get_object_info(self, identifier, variety, **kwargs):
        raise NotImplementedError

    def _get_object_info_where(self, variety, **kwargs):
        raise NotImplementedError

    def get_object_info(self, identifier, variety, decode=True, **kwargs):
        r = self._get_object_info(identifier, variety, **kwargs)
        if r is None:
            raise FileNotFoundError('Object doesn\'t exist')
        if decode:
            r = self.convert_from_bytes_to_types(r)
            r = self._get_file_content(r)
        return r

    def get_object_info_where(self, variety, **kwargs):
        return self._get_object_info_where(variety, **kwargs)

    def get_query_for_validation_set(self, validation_set):
        raise NotImplementedError

    def _get_watcher_for_learning_task(self, learning_task):
        info = self.get_object_info(learning_task, 'learning_task')
        key_to_watch = info['keys_to_watch'][0]
        model_identifier = next(
            m for i, m in enumerate(info['models']) if info['keys'][i] == key_to_watch
        )
        return f'[{learning_task}]:{model_identifier}/{key_to_watch}'

    def _get_watcher_for_neighbourhood(self, identifier):
        return self.get_object_info(identifier, 'neighbourhood')['watcher_identifier']

    def insert(self, insert: Insert, refresh=True, verbose=True):
        for item in insert.documents:
            r = random.random()
            try:
                valid_probability = self.get_meta_data(key='valid_probability')
            except TypeError:
                valid_probability = 0.05
            if '_fold' not in item:
                item['_fold'] = 'valid' if r < valid_probability else 'train'

        output = self._base_insert(insert.to_raw(self.types, self.type_lookup))
        if not refresh:  # pragma: no cover
            return output, None
        task_graph = self._build_task_workflow(
            insert.select_table, ids=output.inserted_ids, verbose=verbose
        )
        task_graph()
        return output, task_graph

    def _insert_validation_data(self, tmp, identifier):
        raise NotImplementedError

    def list_jobs(self):
        """
        List jobs
        """
        raise NotImplementedError

    def list_learning_tasks(self):
        """
        List learning tasks.
        """
        return self.list_components('learning_task')

    def list_metrics(self):
        """
        List metrics.
        """
        return self.list_components('metric')

    def list_models(self):
        """
        List models.
        """
        return self.list_components('model')

    def list_neighbourhoods(self):
        """
        List neighbourhoods.
        """
        return self.list_components('neighbourhood')

    def list_components(self, variety, **kwargs):
        raise NotImplementedError

    def list_postprocessors(self):
        """
        List postprocesors.
        """
        return self.list_components('postprocessor')

    def list_preprocessors(self):
        """
        List preprocessors.
        """
        return self.list_components('preprocessor')

    def list_trainers(self):
        """
        List trainers.
        """
        return self.list_components('trainer')

    def list_types(self):
        """
        List types.
        """
        return self.list_components('type')

    def list_validation_sets(self):
        """
        List validation sets.
        """
        raise NotImplementedError

    def list_watchers(self):
        """
        List watchers.
        """
        return self.list_components('watcher')

    def _load_blob_of_bytes(self, file_id):
        raise NotImplementedError

    def _load_hashes(self, identifier):
        vector_index = self.load_component(identifier, 'vector_index')
        c = self.select(vector_index.watcher.select)
        watcher_info = self.get_object_info(vector_index.watcher.identifier, 'watcher')
        loaded = []
        ids = []
        docs = progress.progressbar(c)
        logging.info(f'loading hashes: "{vector_index.identifier}')
        for r in docs:
            h = self._get_output_from_document(r, watcher_info['key'], watcher_info['model'])
            loaded.append(h)
            ids.append(r['_id'])
        h = vector_index.hash_set_cls(
            loaded,
            ids,
            measure=vector_index.measure,
        )
        self._all_hash_sets[identifier] = h

    def load_component(self, identifier, variety):
        info = self.get_object_info(identifier, variety)
        if info is None:
            raise Exception(
                f'No such object of type "{variety}", '
                f'"{identifier}" has been registered.'
            )
        if 'serializer' not in info:
            info['serializer'] = 'pickle'
        if 'serializer_kwargs' not in info:
            info['serializer_kwargs'] = {}
        m = self._load_serialized_file(info['object'], serializer=info['serializer'])
        m.repopulate(self)
        if variety in self.variety_to_cache_mapping:
            getattr(self, self.variety_to_cache_mapping[variety])[m.identifier] = m
        return m

    def _load_serialized_file(self, file_id, serializer='pickle'):
        bytes_ = self._load_blob_of_bytes(file_id)
        f = io.BytesIO(bytes_)
        if serializer == 'pickle':
            return pickle.load(f)
        elif serializer == 'dill':
            import dill

            return dill.load(f)
        raise NotImplementedError

    @work
    def apply_watcher(
        self,
        identifier,
        ids: List(ObjectId) = None,
        verbose=False,
        max_chunk_size=5000,
        model=None,
        recompute=False,
        watcher_info=None,
        **kwargs,
    ):
        if watcher_info is None:
            watcher_info = self.get_object_info(identifier, 'watcher')
        select = self.select_cls(**watcher_info['select'])
        if ids is None:
            ids = self._get_ids_from_select(select.select_only_id)
        if max_chunk_size is not None:
            for it, i in enumerate(range(0, len(ids), max_chunk_size)):
                logging.info(
                    'computing chunk '
                    f'({it + 1}/{math.ceil(len(ids) / max_chunk_size)})'
                )
                self.apply_watcher(
                    identifier,
                    ids=ids[i : i + max_chunk_size],
                    verbose=verbose,
                    max_chunk_size=None,
                    model=model,
                    recompute=recompute,
                    watcher_info=watcher_info,
                    remote=False,
                    **kwargs,
                )
            return

        model_info = self.get_object_info(watcher_info['model'], 'model')
        outputs = self._compute_model_outputs(
            model_info,
            ids,
            select,
            key=watcher_info['key'],
            features=watcher_info.get('features', {}),
            model=model,
            predict_kwargs=watcher_info.get('predict_kwargs', {}),
        )
        type_ = model_info.get('type')
        if type_ is not None:
            type_ = self.types[type_]
            outputs = [
                {'_content': {'bytes': type_.encode(x), 'type': model_info['type']}}
                for x in outputs
            ]

        self._write_watcher_outputs(watcher_info, outputs, ids)
        return outputs

    def _replace_model(self, identifier, object):
        info = self.get_object_info(identifier, 'model')
        if 'serializer' not in info:
            info['serializer'] = 'pickle'
        if 'serializer_kwargs' not in info:
            info['serializer_kwargs'] = {}
        assert (
            identifier in self.list_models()
        ), f'model "{identifier}" doesn\'t exist to replace'
        file_id = self._create_serialized_file(
            object,
            serializer=info['serializer'],
            serializer_kwargs=info['serializer_kwargs'],
        )
        self._replace_object(info['object'], file_id, 'model', identifier)

    def _replace_object(self, file_id, new_file_id, variety, identifier):
        raise NotImplementedError

    def _save_blob_of_bytes(self, bytes_):
        raise NotImplementedError

    def save_metrics(self, identifier, variety, metrics):
        """
        Save metrics (during learning) into learning-task record.

        :param identifier: identifier of object record
        :param variety: variety of object record
        :param metrics: values of metrics to save
        """
        raise NotImplementedError

    def select(
        self,
        select: Select,
        like=None,
        download=False,
        vector_index=None,
        similar_first=False,
        features=None,
        raw=False,
        n=100,
    ):
        if download and like is not None:
            like = self._get_content_for_filter(like)  # pragma: no cover
        if like is not None:
            if similar_first:
                return self._select_similar_then_matches(
                    like,
                    select,
                    raw=raw,
                    features=features,
                    vector_index=vector_index,
                    n=n,
                )
            else:
                return self._select_matches_then_similar(
                    like,
                    vector_index,
                    select,
                    raw=raw,
                    n=n,
                    features=features,
                )
        else:
            if raw:
                return self._get_raw_cursor(select)
            else:
                return self._get_cursor(select, features=features)

    def _select_matches_then_similar(
        self,
        like,
        vector_index,
        select: Select,
        raw=False,
        n=100,
        features=None,
    ):
        if not select.is_trivial:
            id_cursor = self._get_raw_cursor(select.select_only_id)
            ids = [x['_id'] for x in id_cursor]
            similar_ids, scores = self.select_nearest(
                like,
                ids=ids,
                n=n,
                vector_index=vector_index,
            )
        else:
            similar_ids, scores = self.select_nearest(like, n=n, vector_index=vector_index)

        if raw:
            return self._get_raw_cursor(select.select_using_ids(similar_ids))
        else:
            return self._get_cursor(
                select.select_using_ids(similar_ids),
                features=features,
                scores=dict(zip(similar_ids, scores)),
            )

    def _select_similar_then_matches(
        self,
        like,
        select: Select,
        raw=False,
        n=100,
        features=None,
        vector_index=None,
    ):
        similar_ids, scores = self.select_nearest(like, n=n, vector_index=vector_index)

        if raw:
            return self._get_raw_cursor(select.select_using_ids(similar_ids))
        else:
            return self._get_cursor(
                select.select_using_ids(similar_ids),
                features=features,
                scores=dict(zip(similar_ids, scores)),
            )

    @vector_search
    def select_nearest(
        self,
        like: Convertible(),
        vector_index: str,
        ids=None,
        n=10,
    ) -> Tuple([List(Convertible()), Any]):

        info = self.get_object_info(vector_index, variety='vector_index')
        models = info['models']
        keys = info['keys']
        self._load_hashes(vector_index)
        hash_set = self._all_hash_sets[vector_index]
        if ids is not None:
            hash_set = hash_set[ids]

        if '_id' in like:
            return hash_set.find_nearest_from_id(like['_id'], n=n)

        available_keys = list(like.keys()) + ['_base']
        model, key = next((m, k) for m, k in zip(models, keys) if k in available_keys)
        document = MongoStyleDict(like)
        if '_outputs' not in document:
            document['_outputs'] = {}
        features = info.get('features', {}) or {}

        for subkey in features:
            if subkey not in document:
                continue
            if subkey not in document['_outputs']:
                document['_outputs'][subkey] = {}
            if features[subkey] not in document['_outputs'][subkey]:
                document['_outputs'][subkey][features[subkey]] = self.models[
                    features[subkey]
                ].predict_one(document[subkey])
            document[subkey] = document['_outputs'][subkey][features[subkey]]
        model_input = document[key] if key != '_base' else document

        model = self.models[model]
        h = model.predict_one(model_input)
        return hash_set.find_nearest_from_hash(h, n=n)

    def separate_query_part_from_validation_record(self, r):
        """
        Separate the info in the record after splitting.

        :param r: record
        """
        raise NotImplementedError

    def _set_content_bytes(self, r, key, bytes_):
        raise NotImplementedError

    def set_job_flag(self, identifier, kw):
        """
        Set key-value pair in job record

        :param identifier: id of job
        :param kw: tuple of key-value pair
        """
        return

    @work
    def fit(self, identifier):
        """
        Execute the learning task.

        :param identifier: Identifier of a learning task.
        """

        learning_task: LearningTask = self.load_component(identifier, 'learning_task')

        trainer = learning_task.training_configuration(
            identifier=identifier,
            keys=learning_task.keys,
            model_names=learning_task.models.aslist(),
            models=learning_task.models,
            database_type=self._database_type,
            database_name=self.name,
            select=learning_task.select,
            validation_sets=learning_task.validation_sets,
            metrics={m.identifier: m for m in learning_task.metrics},
            features=learning_task.features,
        )

        try:
            trainer()
        except Exception as e:
            self.delete_component(identifier, 'learning_task', force=True)
            raise e

    def unset_hash_set(self, identifier):
        """
        Remove hash-set from memory

        :param identifier: identifier of corresponding semantic-index
        """
        try:
            del self._all_hash_sets[identifier]
        except KeyError:
            pass

    def update(self, update: Update, refresh=True, verbose=True):
        if refresh and self.list_models():
            ids = self._get_ids_from_select(update.select_ids)
        result = self._base_update(update.to_raw(self.types, self.type_lookup))
        if refresh and self.list_models():
            task_graph = self._build_task_workflow(
                update.select, ids=ids, verbose=verbose
            )
            task_graph()
            return result, task_graph
        return result

    def _update_job_info(self, identifier, key, value):
        raise NotImplementedError

    def _update_neighbourhood(self, ids, similar_ids, identifier, select: Select):
        raise NotImplementedError

    def _update_object_info(self, identifier, variety, key, value):
        raise NotImplementedError

    def validate_semantic_index(self, identifier, validation_sets, metrics):
        """
        Evaluate quality of semantic-index

        :param identifier: identifier of semantic index
        :param validation_sets: validation-sets on which to validate
        :param metrics: metric functions to compute
        """
        results = {}
        for vs in validation_sets:
            results[vs] = validate_representations(self, vs, identifier, metrics)
        for vs in results:
            for m in results[vs]:
                self._update_object_info(
                    identifier,
                    'semantic_index',
                    f'final_metrics.{vs}.{m}',
                    results[vs][m],
                )

    def watch_job(self, identifier):
        """
        Watch stdout/stderr of worker job.

        :param identifier: job-id
        """
        try:
            status = 'pending'
            n_lines = 0
            n_lines_stderr = 0
            while status in {'pending', 'running'}:
                r = self._get_job_info(identifier)
                status = r['status']
                if status == 'running':
                    if len(r['stdout']) > n_lines:
                        print(''.join(r['stdout'][n_lines:]), end='')
                        n_lines = len(r['stdout'])
                    if len(r['stderr']) > n_lines_stderr:
                        print(''.join(r['stderr'][n_lines_stderr:]), end='')
                        n_lines_stderr = len(r['stderr'])
                    time.sleep(0.2)
                else:
                    time.sleep(0.2)
            r = self._get_job_info(identifier)
            if status == 'success':
                if len(r['stdout']) > n_lines:
                    print(''.join(r['stdout'][n_lines:]), end='')
                if len(r['stderr']) > n_lines_stderr:
                    print(''.join(r['stderr'][n_lines_stderr:]), end='')
            elif status == 'failed':  # pragma: no cover
                print(r['msg'])
        except KeyboardInterrupt:  # pragma: no cover
            return

    def write_output_to_job(self, identifier, msg, stream):
        """
        Write stdout/ stderr to database

        :param identifier: job identifier
        :param msg: msg to write
        :param stream: {'stdout', 'stderr'}
        """
        raise NotImplementedError

    def _write_watcher_outputs(self, info, outputs, _ids):
        raise NotImplementedError

    def _base_update(self, update: Update):
        raise NotImplementedError

    def _base_delete(self, delete: Delete):
        raise NotImplementedError

    def _unset_watcher_outputs(self, info):
        raise NotImplementedError
