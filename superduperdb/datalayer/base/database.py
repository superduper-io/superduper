import io
import math
import pickle
import time
from collections import defaultdict
from typing import List

import click
import networkx
from bson import ObjectId

import superduperdb.vector_search.faiss.hashes
import superduperdb.vector_search.vanilla.hashes
import superduperdb.vector_search.vanilla.measures
from superduperdb import cf
from superduperdb.cluster.client_decorators import model_server
from superduperdb.cluster.annotations import Convertible
from superduperdb.cluster.job_submission import work
from superduperdb.cluster.task_workflow import TaskWorkflow
from superduperdb.models.base import wrap_model
from superduperdb.fetchers.downloads import gather_uris
from superduperdb.misc.special_dicts import ArgumentDefaultDict
from superduperdb.fetchers.downloads import Downloader
from superduperdb.misc import progress
from superduperdb.models.vanilla.wrapper import FunctionWrapper
from superduperdb.misc.logger import logging
from superduperdb.vector_search import hash_set_classes
from superduperdb.vector_search.vanilla.hashes import VanillaHashSet


class BaseDatabase:
    """
    Base database connector for SuperDuperDB - all database types should subclass this
    type.
    """
    def __init__(self):

        self.measures = ArgumentDefaultDict(lambda x: self._load_object(x, 'measure'))
        self.metrics = ArgumentDefaultDict(lambda x: self._load_object(x, 'metric'))
        self.models = ArgumentDefaultDict(lambda x: self._load_object(x, 'model'))
        self.types = ArgumentDefaultDict(lambda x: self._load_object(x, 'type'))

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
    def type_lookup(self):
        if self._type_lookup is None:
            self._reload_type_lookup()
        return self._type_lookup

    def _add_split_to_row(self, r, other):
        raise NotImplementedError

    @model_server
    def predict_one(self, model, input_: Convertible(), **kwargs) -> Convertible():
        if isinstance(model, str):
            model = self.models[model]
        return model.predict_one(input_, **{k: v for k, v in kwargs.items() if k != 'remote'})

    @model_server
    def predict(self, model, input_: Convertible(), **kwargs) -> Convertible():
        """
        Apply model to input.

        :param model: model or ``str`` referring to an uploaded model (see ``self.list_models``)
        :param input_: input_ to be passed to the model.
                       Must be possible to encode with registered types (``self.list_types``)
        :param kwargs: key-values (see ``superduperdb.models.utils.predict``)
        """
        if isinstance(model, str):
            model = self.models[model]
        return model.predict(input_, **kwargs)

    def cancel_job(self, job_id):
        raise NotImplementedError

    def _compute_model_outputs(self, model_info, _ids, *query_params, key='_base', features=None,
                               model=None, predict_kwargs=None):

        logging.info('finding documents under filter')
        features = features or {}
        model_identifier = model_info['identifier']
        if features is None:
            features = {}  # pragma: no cover

        documents = self._get_docs_from_ids(_ids, *query_params, features=features)
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
        info = self.get_object_info(identifier, 'neighbourhood')
        watcher_info = self.get_object_info(identifier, 'watcher')
        ids = self._get_ids_from_query(*watcher_info['query_params'])
        logging.info(f'getting hash set')
        h = self._get_hashes_for_query_parameters(info['semantic_index'], *info['query_params'])
        logging.info(f'computing neighbours based on neighbourhood "{identifier}" and '
                     f'index "{info["semantic_index"]}"')

        for i in progress.progressbar(range(0, len(ids), info['batch_size'])):
            sub = ids[i: i + info['batch_size']]
            results = h.find_nearest_from_ids(sub, n=info['n'])
            similar_ids = [res['_ids'] for res in results]
            self._update_neighbourhood(sub, similar_ids, identifier, *info['query_params'])

    def convert_from_bytes_to_types(self, r):
        """
        Convert the bson byte objects in a nested dictionary into python objects.

        :param r: dictionary potentially containing non-Bsonable content
        """
        if isinstance(r, dict) and '_content' in r:
            converter = self.types[r['_content']['type']]
            return converter.decode(r['_content']['bytes'])
        elif isinstance(r, list):
            return [self.convert_from_bytes_to_types(x) for x in r]
        elif isinstance(r, dict):
            for k in r:
                r[k] = self.convert_from_bytes_to_types(r[k])
        return r

    def convert_from_types_to_bytes(self, r):
        """
        Convert the non-Bsonable python objects in a nested dictionary into ``bytes``

        :param r: dictionary potentially containing non-Bsonable content
        """
        if isinstance(r, dict):
            for k in r:
                r[k] = self.convert_from_types_to_bytes(r[k])
            return r
        try:
            t = self.type_lookup[type(r)]
        except KeyError:
            t = None
        if t is not None:
            return {'_content': {'bytes': self.types[t].encode(r), 'type': t}}
        return r

    def _create_job_record(self, *args, **kwargs):
        raise NotImplementedError

    def create_learning_task(self, models, keys, *query_params, identifier=None,
                             configuration=None, verbose=False, validation_sets=(), metrics=(),
                             keys_to_watch=(), features=None, serializer='pickle'):

        if identifier is None:
            identifier = '+'.join([f'{m}/{k}' for m, k in zip(models, keys)])

        assert identifier not in self.list_learning_tasks()

        for k in keys_to_watch:
            assert f'{identifier}/{k}' not in self.list_watchers()

        file_id = self._create_serialized_file(configuration, serializer=serializer)

        self._create_object_entry({
            'variety': 'learning_task',
            'identifier': identifier,
            'query_params': query_params,
            'configuration': {'_file_id': file_id},
            'models': models,
            'keys': keys,
            'metrics': metrics,
            'features': features,
            'validation_sets': validation_sets,
            'keys_to_watch': keys_to_watch,
        })
        model_lookup = dict(zip(keys_to_watch, models))
        jobs = []
        if callable(configuration):
            jobs.append(self.fit(identifier))
        for k in keys_to_watch:
            jobs.append(self._create_watcher(
                f'[{identifier}]:{model_lookup[k]}/{k}',
                model_lookup[k],
                query_params,
                key=k,
                features=features,
                predict_kwargs=configuration.get('loader_kwargs', {}) if configuration is not None else {},
                dependencies=[jobs[0]] if jobs else (),
                verbose=verbose,
            ))
        return jobs

    def create_measure(self, identifier, object, **kwargs):
        """
        Create measure function, called by ``self.create_semantic_index``, to measure similarity
        between model outputs.

        :param identifier: identifier
        :param object: Python object
        """
        return self._create_object(identifier, object, 'measure', **kwargs)

    def create_metric(self, identifier, object, **kwargs):
        """
        Create metric, called by ``self.create_learning_task``, to measure performance of
        learning on validation_sets (see ``self.list_validation_sets``)

        :param identifier: identifier
        :param object: Python object
        """
        return self._create_object(identifier, object, 'metric', **kwargs)

    def create_model(self, identifier, object, preprocessor=None, postprocessor=None, type=None,
                     **kwargs):
        """
        Create a model registered in the collection directly from a python session.
        The added model will then watch incoming records and add outputs computed on those
        records into the ``"_outputs"`` fields of the records.
        The model is then stored inside MongoDB and can be accessed using the ``SuperDuperClient``.

        :param identifier: name of model
        :param object: if specified the model object (pickle-able) else None if model already exists
        :param preprocessor: separate preprocessing
        :param postprocessor: separate postprocessing
        :param type: type for converting model outputs back and forth from bytes
        """

        if identifier.startswith('_'):
            raise ValueError('Identifiers starting with "_" are reserved.')

        assert identifier not in self.list_models()

        if type is not None:
            assert type in self.list_types()

        object = wrap_model(object, preprocessor, postprocessor)

        if isinstance(object, str):
            file_id = object
        else:
            file_id = self._create_serialized_file(object, **kwargs)

        self._create_object_entry({
            'variety': 'model',
            'identifier': identifier,
            'object': file_id,
            'type': type,
            **kwargs,
        })

    def create_neighbourhood(self, identifier, n=10, batch_size=100):
        """
        Cache similarity between items of model watcher (see ``self.list_watchers``)

        :param identifier: identifier of watcher to use
        :param n: number of similar items
        :param batch_size: batch_size used in computation
        """
        assert identifier in self.list_watchers()
        assert identifier not in self.list_neighbourhoods()
        self._create_object_entry({
            'identifier': identifier,
            'n': n,
            'batch_size': batch_size,
            'variety': 'neighbourhood',
        })
        return self.compute_neighbourhood(identifier)

    def _create_object_entry(self, info):
        raise NotImplementedError

    def _create_object(self, identifier, object, variety, serializer='pickle', serializer_kwargs=None):
        serializer_kwargs = serializer_kwargs or {}
        assert identifier not in self._list_objects(variety)
        secrets = {}
        file_id = self._create_serialized_file(object, serializer=serializer,
                                               serializer_kwargs=serializer_kwargs)
        self._create_object_entry({
            'identifier': identifier,
            'object': file_id,
            'variety': variety,
            'serializer': serializer,
            'serializer_kwargs': serializer_kwargs,
            **secrets,
        })

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

    def _create_serialized_file(self, object, serializer='pickle', serializer_kwargs=None):
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

    def create_type(self, identifier, object, **kwargs):
        """
        Create datatype, in order to serialize python object data in the database.

        :param identifier: identifier
        :param object: Python object
        """
        return self._create_object(identifier, object, 'type', **kwargs)

    def _create_validation_set(self, identifier, *query_params, chunk_size=1000):
        if identifier in self.list_validation_sets():
            raise Exception(f'validation set {identifier} already exists!')
        data = self.execute_query(*query_params)
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

    def _create_watcher(self, identifier, model, query_params, key='_base', verbose=False,
                        target=None, process_docs=True, features=None, predict_kwargs=None,
                        dependencies=()):

        assert identifier not in self.list_watchers()

        self._create_object_entry({
            'identifier': identifier,
            'variety': 'watcher',
            'model': model,
            'query_params': query_params,
            'key': key,
            'features': features if features else {},
            'target': target,
            'predict_kwargs': predict_kwargs or {},
        })

        if process_docs:
            ids = self._get_ids_from_query(*query_params)
            if not ids:
                return
            return self.apply_watcher(
                identifier,
                ids=ids,
                verbose=verbose,
                dependencies=dependencies,
                remote=self.remote,
            )

    def delete_learning_task(self, identifier, force=False):
        """
        Delete function.

        :param name: name of function
        :param force: toggle to ``True`` to skip confirmation step
        """
        do_delete = False
        if force or click.confirm(f'Are you sure you want to delete the learning-task "{identifier}"?',
                                  default=False):
            do_delete = True
        if not do_delete:
            return

        info = self.get_object_info(identifier, 'learning_task')
        if info is None and force:
            return

        model_lookup = dict(zip(info['keys'], info['models']))
        for k in info['keys_to_watch']:
            self._delete_object_info(f'[{identifier}]:{model_lookup[k]}/{k}', 'watcher')
        self._delete_object_info(info['identifier'], 'learning_task')

    def delete_measure(self, identifier, force=False):
        """
        Delete measure.

        :param identifier: identifier of measure
        :param force: toggle to ``True`` to skip confirmation step
        """
        return self._delete_object(identifier, 'measure', force=force)

    def delete_metric(self, identifier, force=False):
        """
        Delete metric.

        :param identifier: identifier of metric
        :param force: toggle to ``True`` to skip confirmation step
        """
        return self._delete_object(identifier, 'metric', force=force)

    def delete_model(self, identifier, force=False):
        """
        Delete model.

        :param identifier: identifier of model
        :param force: toggle to ``True`` to skip confirmation step
        """
        return self._delete_object(identifier, 'model', force=force)

    def delete_neighbourhood(self, identifier, force=False):
        """
        Delete neighbourhood.

        :param name: name of neighbourhood
        :param force: toggle to ``True`` to skip confirmation step
        """
        info = self.get_object_info(identifier, 'neighbourhood')
        watcher_info = self.get_object_info(info['watcher_identifier'], 'watcher')
        if force or click.confirm(f'Removing neighbourhood "{identifier}"'
                                  ' documents. Are you sure?', default=False):
            self._unset_neighbourhood_data(info, watcher_info)
            self._delete_object_info(identifier, 'neighbourhood')
        else:
            logging.info('aborting') # pragma: no cover

    def _delete_object(self, identifier, variety, force=False):
        info = self.get_object_info(identifier, variety)
        if not info:
            if not force:
                raise Exception(f'"{identifier}": {variety} does not exist...')
            return
        if force or click.confirm(f'You are about to delete {variety}: {identifier}, are you sure?',
                                  default=False):
            if hasattr(self, variety + 's') and identifier in getattr(self, variety + 's'):
                del getattr(self, variety + 's')[identifier]
            self.filesystem.delete(info['object'])
            self._delete_object_info(identifier, variety)

    def _delete_object_info(self, identifier, variety):
        raise NotImplementedError

    def delete_type(self, identifier, force=False):
        """
        Delete type.

        :param identifier: identifier of type
        :param force: toggle to ``True`` to skip confirmation step
        """
        return self._delete_object(identifier, 'type', force=force)

    def delete_watcher(self, identifier, force=False, delete_outputs=True):
        """
        Delete watcher

        :param name: Name of model
        :param force: Toggle to ``True`` to skip confirmation
        """
        info = self.get_object_info(identifier, 'watcher')
        do_delete = False
        if force or click.confirm(f'Are you sure you want to delete this watcher: {identifier}; ',
                                  default=False):
            do_delete = True
        if not do_delete:
            return

        if info.get('target') is None and delete_outputs:
            self._unset_watcher_outputs(info)
        self._delete_object_info(identifier, 'watcher')

    @work
    def download_content(self, query_params, ids=None, documents=None, timeout=None,
                         raises=True, n_download_workers=None, headers=None, **kwargs):
        logging.debug(query_params)
        logging.debug(ids)
        update_db = False
        if documents is None:
            update_db = True
            if ids is None:
                documents = list(self.execute_query(*query_params))
            else:
                documents = self._get_docs_from_ids(ids, *query_params, raw=True)
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

        table = self._get_table_from_query_params(query_params)
        downloader = Downloader(
            table=table,
            uris=uris,
            ids=place_ids,
            keys=keys,
            update_one=self._download_update,
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

    def _download_update(self, table, _id, key, bytes_):
        raise NotImplementedError

    def execute_query(self, *args, **kwargs):
        raise NotImplementedError

    def _format_fold_to_query(self, query_params, fold):
        raise NotImplementedError

    def _get_content_for_filter(self, filter):
        if '_id' not in filter:
            filter['_id'] = 0
        uris = gather_uris([filter])[0]
        if uris:
            filter = self.download_content(self.name,
                                           documents=[filter],
                                           timeout=None, raises=True)[0]
            filter = self.convert_from_bytes_to_types(filter)
        return filter

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

    def _get_docs_from_ids(self, ids, *query_params, features=None, raw=False):
        raise NotImplementedError

    def _get_file_content(self, r):
        for k in r:
            if isinstance(r[k], dict):
                if '_file_id' in r[k]:
                    r[k] = self._load_serialized_file(r[k]['_file_id'])
                else:
                    r[k] = self._get_file_content(r[k])
        return r

    def _get_hash_from_record(self, r, watcher_info):
        raise NotImplementedError

    def _get_hashes_for_query_parameters(self, semantic_index, *query_params):
        raise NotImplementedError

    def _get_job_info(self, identifier):
        raise NotImplementedError

    def get_ids_from_result(self, query_params, result):
        raise NotImplementedError

    def _get_ids_from_query(self, *query_params):
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

    def get_query_params_for_validation_set(self, validation_set):
        raise NotImplementedError

    def _get_watcher_for_neighbourhood(self, identifier):
        return self.get_object_info(identifier, 'neighbourhood')['watcher_identifier']

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
        return self._list_objects('learning_task')

    def list_measures(self):
        """
        List measures.
        """
        return self._list_objects('measure')

    def list_metrics(self):
        """
        List metrics.
        """
        return self._list_objects('metric')

    def list_models(self):
        """
        List models.
        """
        return self._list_objects('model')

    def list_neighbourhoods(self):
        """
        List neighbourhoods.
        """
        return self._list_objects('neighbourhood')

    def _list_objects(self, variety, **kwargs):
        raise NotImplementedError

    def list_postprocessors(self):
        """
        List postprocesors.
        """
        return self._list_objects('postprocessor')

    def list_preprocessors(self):
        """
        List preprocessors.
        """
        return self._list_objects('preprocessor')

    def list_trainers(self):
        """
        List trainers.
        """
        return self._list_objects('trainer')

    def list_types(self):
        """
        List types.
        """
        return self._list_objects('type')

    def list_validation_sets(self):
        """
        List validation sets.
        """
        raise NotImplementedError

    def list_watchers(self):
        """
        List watchers.
        """
        return self._list_objects('watcher')

    def _load_blob_of_bytes(self, file_id):
        raise NotImplementedError

    def _load_hashes(self, watcher, measure='css', hash_set_cls='vanilla'):
        watcher_info = self.get_object_info(watcher, 'watcher')
        filter = watcher_info.get('filter', {})
        key = watcher_info.get('key', '_base')
        filter[f'_outputs.{key}.{watcher_info["model"]}'] = {'$exists': 1}
        c = self.execute_query(*watcher_info['query_params'])
        loaded = []
        ids = []
        docs = progress.progressbar(c)
        logging.info(f'loading hashes: "{watcher_info["identifier"]}"')
        for r in docs:
            h = self._get_hash_from_record(r, watcher_info)
            loaded.append(h)
            ids.append(r['_id'])
        if hash_set_cls is None:
            hash_set_cls = VanillaHashSet
        hash_set_cls = hash_set_classes[hash_set_cls]
        h = hash_set_cls(
            loaded,
            ids,
            measure=measure,
        )
        self._all_hash_sets[watcher] = h

    def _get_watcher_for_learning_task(self, learning_task):
        info = self.get_object_info(learning_task, 'learning_task')
        key_to_watch = info['keys_to_watch'][0]
        model_identifier = next(
            m for i, m in enumerate(info['models']) if info['keys'][i] == key_to_watch)
        return f'[{learning_task}]:{model_identifier}/{key_to_watch}'

    def _load_hashes_from_watcher_info(self):
        ...

    def _load_hashes_from_learning_task(self, identifier):
        info = self.get_object_info(identifier, 'learning_task')
        key_to_watch = info['keys_to_watch'][0]
        model_identifier = next(m for i, m in enumerate(info['models']) if info['keys'][i] == key_to_watch)
        watcher_info = self.get_object_info(f'[{identifier}]:{model_identifier}/{key_to_watch}', 'watcher')
        filter = watcher_info.get('filter', {})
        key = watcher_info.get('key', '_base')
        filter[f'_outputs.{key}.{watcher_info["model"]}'] = {'$exists': 1}
        c = self.execute_query(*watcher_info['query_params'])

        configuration = info.get('configuration', {}) or {}
        loaded = []
        ids = []
        docs = progress.progressbar(c)
        logging.info(f'loading hashes: "{identifier}"')
        for r in docs:
            h = self._get_hash_from_record(r, watcher_info)
            loaded.append(h)
            ids.append(r['_id'])

        hash_set_cls = configuration.get(
            'hash_set_cls',
             superduperdb.vector_search.vanilla.hashes.VanillaHashSet,
        )
        return hash_set_cls(
            loaded,
            ids,
            measure=configuration.get('measure', superduperdb.vector_search.vanilla.measures.css),
        )

    def _load_object(self, identifier, variety):
        info = self.get_object_info(identifier, variety)
        if info is None:
            raise Exception(f'No such object of type "{variety}", '
                            f'"{identifier}" has been registered.')  # pragma: no cover
        if 'serializer' not in info:
            info['serializer'] = 'pickle'
        if 'serializer_kwargs' not in info:
            info['serializer_kwargs'] = {}
        m = self._load_serialized_file(info['object'], serializer=info['serializer'])
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
    def apply_watcher(self, identifier, ids: List[ObjectId] = None, verbose=False,
                      max_chunk_size=5000, model=None, recompute=False, watcher_info=None,
                      **kwargs):

        if watcher_info is None:
            watcher_info = self.get_object_info(identifier, 'watcher')
        if ids is None:
            ids = self._get_ids_from_query(*watcher_info['query_params'])
        if max_chunk_size is not None:
            for it, i in enumerate(range(0, len(ids), max_chunk_size)):
                logging.info('computing chunk '
                             f'({it + 1}/{math.ceil(len(ids) / max_chunk_size)})')
                self.apply_watcher(
                    identifier,
                    ids=ids[i: i + max_chunk_size],
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
        outputs = self._compute_model_outputs(model_info,
                                              ids,
                                              *watcher_info['query_params'],
                                              key=watcher_info['key'],
                                              features=watcher_info.get('features', {}),
                                              model=model,
                                              predict_kwargs=watcher_info.get('predict_kwargs', {}))
        type_ = model_info.get('type')
        if type_ is not None:
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

        self._write_watcher_outputs(watcher_info, outputs, ids)
        return outputs

    def _build_task_workflow(self, query_params, ids=None, dependencies=(), verbose=True):
        job_ids = defaultdict(lambda: [])
        job_ids.update(dependencies)
        G = TaskWorkflow(self)
        if ids is None:
            ids = self._get_ids_from_query(query_params)

        G.add_node(
            f'{self.download_content.__name__}()',
            data={
                'task': self.download_content,
                'args': [
                    query_params,
                ],
                'kwargs': {
                    'ids': ids,
                }
            }
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
                    }
                }
            )

        for identifier in self.list_neighbourhoods():
            G.add_node(
                f'{self.compute_neighbourhood.__name__}({identifier})',
                data={
                    'task': self.compute_neighbourhood,
                    'args': [identifier],
                    'kwargs': {}
                }
            )

        for identifier in self.list_watchers():
            G.add_edge(
                f'{self.download_content.__name__}()',
                f'{self.apply_watcher.__name__}({identifier})'
            )
            deps = self._get_dependencies_for_watcher(identifier)
            for dep in deps:
                G.add_edge(
                    f'{self.apply_watcher.__name__}({dep})',
                    f'{self.apply_watcher.__name__}({identifier})'
                )
                G.add_edge(
                    f'{self.download_content.__name__}()',
                    f'{self.apply_watcher.__name__}({identifier})'
                )

        for identifier in self.list_neighbourhoods():
            watcher_identifier = self._get_watcher_for_neighbourhood(identifier)
            G.add_edge(('watcher', watcher_identifier), ('neighbourhood', identifier))
            G.add_edge(
                f'{self.apply_watcher.__name__}({watcher_identifier})',
                f'{self.compute_neighbourhood.__name__}({identifier})'
            )
        return G

    def _replace_model(self, identifier, object):
        info = self.get_object_info(identifier, 'model')
        if 'serializer' not in info:
            info['serializer'] = 'pickle'
        if 'serializer_kwargs' not in info:
            info['serializer_kwargs'] = {}
        assert identifier in self.list_models(), f'model "{identifier}" doesn\'t exist to replace'
        file_id = self._create_serialized_file(object,
                                               serializer=info['serializer'],
                                               serializer_kwargs=info['serializer_kwargs'])
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
        raise NotImplementedError

    def _get_table_from_query_params(self, query_params):
        raise NotImplementedError

    @work
    def fit(self, identifier):

        info = self.get_object_info(identifier, 'learning_task')
        trainer = info['configuration'](
            identifier=identifier,
            models=[
                self.models[m] if m != '_identity' else FunctionWrapper(lambda x: x)
                for m in info['models']
            ],
            keys=info['keys'],
            model_names=info['models'],
            database_type=self._database_type,
            database_name=self.name,
            query_params=info['query_params'],
            validation_sets=info['validation_sets'],
            metrics={k: self.metrics[k] for k in info['metrics']},
            features=info['features'],
        )
        try:
            trainer()
        except Exception as e:
            self.delete_learning_task(identifier, force=True)
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

    def _update_job_info(self, identifier, key, value):
        raise NotImplementedError

    def _update_neighbourhood(self, ids, similar_ids, identifier, *query_params):
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
                    identifier, 'semantic_index',
                    f'final_metrics.{vs}.{m}', results[vs][m],
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
            elif status == 'failed': # pragma: no cover
                print(r['msg'])
        except KeyboardInterrupt: # pragma: no cover
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


