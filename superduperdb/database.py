import math
import multiprocessing
import os
import time
from collections import defaultdict

import click
import networkx
import torch

from superduperdb import cf, training
from superduperdb.getters import jobs
from superduperdb.lookup import hashes
from superduperdb.training.validation import validate_representations
from superduperdb.types.utils import convert_from_bytes_to_types
from superduperdb.utils import gather_urls
from superduperdb.getters import client as our_client
from superduperdb.models.utils import BasicDataset, create_container, Container, apply_model
from superduperdb.utils import ArgumentDefaultDict, progressbar, unpack_batch, Downloader


class BaseDatabase:
    def __init__(self):

        self.models = ArgumentDefaultDict(lambda x: self._load_model(x))
        self.functions = ArgumentDefaultDict(lambda x: self._load_object(x, 'function'))
        self.preprocessors = ArgumentDefaultDict(lambda x: self._load_object(x, 'preprocessor'))
        self.postprocessors = ArgumentDefaultDict(lambda x: self._load_object(x, 'postprocessor'))
        self.types = ArgumentDefaultDict(lambda x: self._load_object(x, 'type'))
        self.splitters = ArgumentDefaultDict(lambda x: self._load_object(x, 'splitter'))
        self.objectives = ArgumentDefaultDict(lambda x: self._load_object(x, 'objective'))
        self.measures = ArgumentDefaultDict(lambda x: self._load_object(x, 'measure'))
        self.metrics = ArgumentDefaultDict(lambda x: self._load_object(x, 'metric'))

        self.remote = cf.get('remote', False)
        self._type_lookup = None

        self._hash_set = None
        self._all_hash_sets = ArgumentDefaultDict(self._load_hashes)

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

    def _add_split_to_row(self, r, other):
        raise NotImplementedError

    def apply_model(self, model, input_, **kwargs):
        if self.remote:
            return our_client.apply_model(self.name, model, input_, **kwargs)
        if isinstance(model, str):
            model = self.models[model]
        with torch.no_grad():
            return apply_model(model, input_, **kwargs)

    def _compute_model_outputs(self, model_info, _ids, *query_params, key='_base', features=None,
                               model=None, verbose=True, loader_kwargs=None):

        print('finding documents under filter')
        features = features or {}
        model_identifier = model_info['identifier']
        if features is None:
            features = {}  # pragma: no cover

        documents = self._get_docs_from_ids(_ids, *query_params, features=features)
        print('done.')
        if key != '_base' or '_base' in features:
            passed_docs = [r[key] for r in documents]
        else:  # pragma: no cover
            passed_docs = documents
        if model is None:  # model is not None during training, since a suboptimal model may be in need of validation
            model = self.models[model_identifier]
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
                print(f'processing with {model_identifier}')
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
        return outputs

    def _compute_neighbourhood(self, identifier):
        import sys
        sys.path.insert(0, os.getcwd())

        info = self.get_object_info(identifier, 'neighbourhood')
        watcher_info = self.get_object_info(identifier, 'watcher')
        ids = self._get_ids_from_query(*watcher_info['query_params'])
        print('getting hash set')
        h = self._get_hashes_for_query_parameters(info['semantic_index'], *info['query_params'])
        print(f'computing neighbours based on neighbourhood "{identifier}" and '
              f'index "{info["semantic_index"]}"')

        for i in progressbar(range(0, len(ids), info['batch_size'])):
            sub = ids[i: i + info['batch_size']]
            results = h.find_nearest_from_ids(sub, n=info['n'])
            similar_ids = [res['_ids'] for res in results]
            self._update_neighbourhood(sub, similar_ids, identifier, *info['query_params'])

    def convert_from_types_to_bytes(self, r):
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

    def create_function(self, name, object):
        return self._create_object(name, object, 'function')

    def _create_imputation(self, identifier, model, model_key, target, target_key, query_params,
                           objective=None, metrics=None,
                           splitter=None, watch=True, loader_kwargs=None, **trainer_kwargs):
        """
        Create an imputation setup. This is any learning task where we have an input to the model
        compared to the target.

        :param identifier: Name of imputation
        :param model: Model settings or model name
        :param model_key: Key for model to injest
        :param target: Target settings (input to ``create_model``) or target name
        :param target_key: Key for model to predict
        :param query_params: Query parameters
        :param objective: Loss settings or objective
        :param metrics: List of metric settings or metric names
        :param splitter: Splitter name to use to prepare data points for insertion to model
        :param loader_kwargs: Keyword-arguments for the watcher
        :param trainer_kwargs: Keyword-arguments to forward to ``train_tools.ImputationTrainer``
        :return: job_ids of jobs required to create the imputation
        """

        assert identifier not in self.list_imputations()
        assert identifier not in self.list_watchers()
        assert target in self.list_functions() or target in self.list_models()
        assert model in self.list_models()
        if objective is not None:
            assert objective in self.list_objectives()
        if metrics:
            for metric in metrics:
                assert metric in self.list_metrics()

        self._create_object_entry({
            'variety': 'imputation',
            'identifier': identifier,
            'model': model,
            'model_key': model_key,
            'target': target,
            'target_key': target_key,
            'query_params': query_params,
            'metrics': metrics or [],
            'objective': objective,
            'splitter': splitter,
            'loader_kwargs': loader_kwargs,
            'trainer_kwargs': trainer_kwargs,
        })

        if objective is None:
            return

        try:
            jobs = [self._train_imputation(identifier)]
        except KeyboardInterrupt:
            print('aborting training early...')
            jobs = []
        if watch:
            jobs.append(self._create_watcher(
                identifier,
                model,
                query_params,
                key=model_key,
                features=trainer_kwargs.get('features', {}),
                dependencies=jobs,
                verbose=True,
            ))
        return jobs

    def create_measure(self, name, object):
        return self._create_object(name, object, 'measure')

    def create_metric(self, name, object):
        return self._create_object(name, object, 'metric')

    def create_model(self, identifier, object=None, preprocessor=None,
                     postprocessor=None, type=None):
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

        assert identifier not in self.list_models()

        if type is not None:
            assert type in self.list_types()

        if isinstance(object, str):
            file_id = object
        else:
            file_id = self._create_pickled_file(object)

        self._create_object_entry({
            'variety': 'model',
            'identifier': identifier,
            'object': file_id,
            'type': type,
            'preprocessor': preprocessor,
            'postprocessor': postprocessor,
        })

    def create_neighbourhood(self, identifier, n=10,
                             batch_size=100):
        assert identifier in self.list_watchers()
        assert identifier not in self.list_neighbourhoods()
        self._create_object_entry({
            'identifier': identifier,
            'n': n,
            'batch_size': batch_size,
            'variety': 'neighbourhood',
        })
        if not self.remote:
            self._compute_neighbourhood(identifier)
        else:
            return jobs.process(
                self.name,
                '_compute_neighbourhood',
                identifier,
            )

    def _create_object_entry(self, info):
        raise NotImplementedError

    def _create_object(self, identifier, object, variety):
        assert identifier not in self._list_objects(variety)
        file_id = self._create_pickled_file(object)
        self._create_object_entry({
            'identifier': identifier,
            'object': file_id,
            'variety': variety,
        })

    def create_objective(self, name, object):
        return self._create_object(name, object, 'objective')

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

    def _create_pickled_file(self, object):
        raise NotImplementedError

    def create_postprocessor(self, name, object):
        return self._create_object(name, object, 'postprocessor')

    def create_preprocessor(self, name, object):
        return self._create_object(name, object, 'preprocessor')

    def _create_semantic_index(self, identifier, query_params, models, keys, measure,
                               validation_sets=(), metrics=(), objective=None,
                               splitter=None, loader_kwargs=None, **trainer_kwargs):
        """
        :param identifier: Name/ unique id to assign to index
        :param models: List of existing models
        :param keys: Keys in incoming data to listen to
        :param measure: Measure name
        :param query_params: How to specify the data (collection/filter or SQL query...)
        :param validation_sets: Name of immutable validation set to be used to evaluate performance
        :param metrics: List of existing metrics,
        :param objective: Loss name
        :param filter_: Filter on which to train
        :param splitter: Splitter name
        :param loader_kwargs: Keyword arguments to be passed to
        :param trainer_kwargs: Keyword arguments to be passed to ``training.train_tools.RepresentationTrainer``
        :return: List of job identifiers if ``self.remote``
        """
        assert identifier not in self.list_semantic_indexes()
        assert identifier not in self.list_watchers()

        if objective is not None:  # pragma: no cover
            if len(models) == 1:
                assert splitter is not None, 'need a splitter for self-supervised ranking...'

        self._create_object_entry({
            'variety': 'semantic_index',
            'identifier': identifier,
            'query_params': query_params,
            'models': models,
            'keys': keys,
            'metrics': metrics,
            'objective': objective,
            'measure': measure,
            'splitter': splitter,
            'validation_sets': validation_sets,
            'trainer_kwargs': trainer_kwargs,
        })

        for vs in validation_sets:
            tmp_query_params = self.get_query_params_for_validation_set(vs)
            self._create_semantic_index(
                identifier + f'/{vs}',
                tmp_query_params,
                models,
                keys,
                measure,
                loader_kwargs=loader_kwargs,
            )

        self._create_watcher(identifier, models[0], query_params, key=keys[0],
                             process_docs=False, features=trainer_kwargs.get('features', {}),
                             loader_kwargs=loader_kwargs or {})
        if objective is None:
            return [self.refresh_watcher(identifier, dependencies=())]
        try:
            jobs = [self._train_semantic_index(identifier)]
        except KeyboardInterrupt:
            print('training aborted...')
            jobs = []
        jobs.append(self.refresh_watcher(identifier, dependencies=jobs))
        return jobs

    def create_splitter(self, name, object):
        return self._create_object(name, object, 'splitter')

    def create_type(self, name, object):
        return self._create_object(name, object, 'type')

    def _create_validation_set(self, identifier, *query_params, chunk_size=1000, splitter=None):
        if identifier in self.list_validation_sets():
            raise Exception(f'validation set {identifier} already exists!')
        if isinstance(splitter, str):
            splitter = self.splitters[splitter]

        data = self.execute_query(*query_params)
        it = 0
        tmp = []
        for r in progressbar(data):
            if splitter is not None:
                r, other = splitter(r)
                r = self._add_split_to_row(r, other)
            tmp.append(r)
            it += 1
            if it % chunk_size == 0:
                self._insert_validation_data(tmp, identifier)
                tmp = []
        if tmp:
            self._insert_validation_data(tmp, identifier)

    def _create_watcher(self, identifier, model, query_params, key='_base', verbose=False,
                        target=None, process_docs=True, features=None, loader_kwargs=None,
                        dependencies=()):

        r = self.get_object_info(identifier, 'watcher')
        if r is not None:
            raise Exception(f'Watcher {identifier} already exists!')

        self._create_object_entry({
            'identifier': identifier,
            'variety': 'watcher',
            'model': model,
            'query_params': query_params,
            'key': key,
            'features': features if features else {},
            'target': target,
            'loader_kwargs': loader_kwargs or {},
        })

        if process_docs:
            ids = self._get_ids_from_query(*query_params)
            if not ids:
                return
            if not self.remote:
                self._process_documents_with_watcher(identifier, ids, verbose=verbose)
            else:  # pragma: no cover
                return jobs.process(
                    self.name,
                    '_process_documents_with_watcher',
                    identifier,
                    ids,
                    verbose=verbose,
                    dependencies=dependencies,
                )

    def delete_function(self, name, force=False):
        return self._delete_object('function', name, force=force)

    def delete_imputation(self, identifier, force=False):
        """
        Delete imputation from collection

        :param identifier: Identifier of imputation
        :param force: Toggle to ``True`` to skip confirmation
        """
        do_delete = False
        if force or click.confirm(f'Are you sure you want to delete the imputation "{identifier}"?',
                                  default=False):
            do_delete = True
        if not do_delete:
            return

        info = self.get_object_info(identifier, 'imputation')
        if info is None and force:
            return

        self._delete_object_info(info['watcher_identifier'], 'watcher')
        self._delete_object_info(info['identifier'], 'imputation')

    def delete_measure(self, name, force=False):
        return self._delete_object(name, ['measure'], force=force)

    def delete_metric(self, name, force=False):
        return self._delete_object(name, ['metric'], force=force)

    def delete_model(self, name, force=False):
        return self._delete_object('model', name, force=force)

    def delete_neighbourhood(self, identifier, force=False):
        info = self.get_object_info(identifier, 'neighbourhood')
        watcher_info = self.get_object_info(info['watcher_identifier'], 'watcher')
        if force or click.confirm(f'Removing neighbourhood "{identifier}"'
                                  ' documents. Are you sure?', default=False):
            self._unset_neighbourhood_data(info, watcher_info)
            self._delete_object_info(identifier, 'neighbourhood')
        else:
            print('aborting') # pragma: no cover

    def _delete_object(self, identifier, variety, force=False):
        info = self.get_object_info(identifier, variety)
        if not info:
            if not force:
                raise Exception(f'{variety} "{identifier}" does not exist...')
            return
        if force or click.confirm(f'You are about to delete {variety}: {identifier}, are you sure?',
                                  default=False):
            self.filesystem.delete(info['object'])
            self._delete_object_info(identifier, variety)

    def _delete_object_info(self, identifier, variety):
        raise NotImplementedError

    def delete_objective(self, name, force=False):
        return self._delete_object(name, 'objective', force=force)

    def delete_postprocessor(self, name, force=False):
        return self._delete_object(name, 'postprocessor', force=force)

    def delete_preprocessor(self, name, force=False):
        return self._delete_object(name, 'preprocessor', force=force)

    def delete_semantic_index(self, identifier, force=False):
        info = self.get_object_info(identifier, 'semantic_index')
        watcher_info = self.get_object_info(identifier, 'watcher')
        if info is None:  # pragma: no cover
            return
        do_delete = False
        if force or \
                click.confirm(f'Are you sure you want to delete this semantic index: "{identifier}"; '):
            do_delete = True

        if not do_delete:
            return

        if watcher_info:
            self.delete_watcher(identifier, force=True)
        self._delete_object_info(identifier, 'semantic_index')

    def delete_splitter(self, name, force=False):
        return self._delete_object(name, ['splitter'], force=force)

    def delete_type(self, name, force=False):
        return self._delete_object(name, ['type'], force=force)

    def delete_watcher(self, identifier, force=False, delete_outputs=True):
        """
        Delete model from collection

        :param collection: Collection name
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

    def _download_content(self, table, *query_params, ids=None, documents=None, timeout=None,
                          raises=True, n_download_workers=None, headers=None):
        import sys
        sys.path.insert(0, os.getcwd())

        update_db = False
        if documents is None:
            update_db = True
            if ids is None:
                documents = list(self.execute_query(*query_params))
            else:
                documents = self._get_docs_from_ids(ids, *query_params, raw=True)
        urls, keys, place_ids = gather_urls(documents)
        print(f'found {len(urls)} urls')
        if not urls:
            return

        if n_download_workers is None:
            try:
                n_download_workers = self.get_meta_data('n_download_workers')
            except TypeError:
                n_download_workers = 0

        if headers is None:
            try:
                headers = self.get_meta_data('headers')
            except TypeError:
                headers = 0

        if timeout is None:
            try:
                timeout = self.get_meta_data('download_timeout')
            except TypeError:
                timeout = None

        downloader = Downloader(
            table=table,
            urls=urls,
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
        # query_params = (collection, filter, [projection])
        query_params = list(query_params)
        query_params[1]['_fold'] = fold
        return tuple(query_params)

    def _get_content_for_filter(self, filter):
        if '_id' not in filter:
            filter['_id'] = 0
        urls = gather_urls([filter])[0]
        if urls:
            filter = self._download_content(self.name,
                                            documents=[filter],
                                            timeout=None, raises=True)[0]
            filter = convert_from_bytes_to_types(filter, converters=self.types)
        return filter

    def _get_dependencies_for_watcher(self, identifier):
        info = self.get_object_info(identifier, 'watcher')
        if info is None:
            return []
        watcher_features = info.get('features', {})
        return list(zip(watcher_features.values(), watcher_features.keys()))

    def _get_docs_from_ids(self, ids, *query_params, features=None, raw=False):
        raise NotImplementedError

    def _get_hash_from_record(self, r, watcher_info):
        raise NotImplementedError

    def _get_hashes_for_query_parameters(self, semantic_index, *query_params):
        raise NotImplementedError

    def _get_hash_set(self, semantic_index):
        return self._all_hash_sets[semantic_index]

    def _get_job_info(self, identifier):
        raise NotImplementedError

    def get_ids_from_result(self, query_params, result):
        raise NotImplementedError

    def _get_ids_from_query(self, *query_params):
        raise NotImplementedError

    def get_meta_data(self, **kwargs):
        raise NotImplementedError

    def get_object_info(self, identifier, variety):
        raise NotImplementedError

    def get_query_params_for_validation_set(self, validation_set):
        raise NotImplementedError

    def _get_watcher_for_neighbourhood(self, identifier):
        return self.get_object_info(identifier, 'neighbourhood')['watcher_identifier']

    def _insert_validation_data(self, tmp, identifier):
        raise NotImplementedError

    def list_functions(self):
        return self._list_objects('function')

    def list_imputations(self):
        return self._list_objects('imputation')

    def list_jobs(self):
        return self._list_objects('jobs')

    def list_measures(self):
        return self._list_objects('measure')

    def list_metrics(self):
        return self._list_objects('metric')

    def list_models(self):
        return self._list_objects('model')

    def list_neighbourhoods(self):
        return self._list_objects('neighbourhood')

    def _list_objects(self, variety):
        raise NotImplementedError

    def list_objectives(self):
        return self._list_objects('objective')

    def list_splitters(self):
        return self._list_objects('splitter')

    def list_postprocessors(self):
        return self._list_objects('postprocessor')

    def list_preprocessors(self):
        return self._list_objects('preprocessor')

    def list_semantic_indexes(self):
        return self._list_objects('semantic_index')

    def list_types(self):
        return self._list_objects('type')

    def list_validation_sets(self):
        raise NotImplementedError

    def list_watchers(self):
        return self._list_objects('watcher')

    def _load_hashes(self, identifier):
        info = self.get_object_info(identifier, 'semantic_index')
        watcher_info = self.get_object_info(identifier, 'watcher')
        filter = watcher_info.get('filter', {})
        key = watcher_info.get('key', '_base')
        filter[f'_outputs.{key}.{watcher_info["model"]}'] = {'$exists': 1}
        c = self.execute_query(*watcher_info['query_params'])
        measure = self.measures[info['measure']]
        loaded = []
        ids = []
        docs = progressbar(c)
        print(f'loading hashes: "{identifier}"')
        for r in docs:
            h = self._get_hash_from_record(r, watcher_info)
            loaded.append(h)
            ids.append(r['_id'])
        return hashes.HashSet(torch.stack(loaded), ids, measure=measure)

    def _load_model(self, identifier):
        info = self.get_object_info(identifier, 'model')
        if info is None:
            raise Exception(f'No such object of type "model", "{identifier}" has been registered.') # pragma: no cover
        info = dict(info)
        model = self._load_pickled_file(info['object'])
        model.eval()

        if info.get('preprocessor') is None and info.get('postprocessor') is None:
            return model

        preprocessor = None
        if info.get('preprocessor') is not None:
            preprocessor = self._load_object('preprocessor', info['preprocessor'])
        postprocessor = None
        if info.get('postprocessor') is not None:
            postprocessor = self._load_object('postprocessor', info['postprocessor'])
        return create_container(preprocessor, model, postprocessor)

    def _load_object(self, identifier, variety):
        info = self.get_object_info(identifier, variety)
        if info is None:
            raise Exception(f'No such object of type "{variety}", '
                            f'"{identifier}" has been registered.')  # pragma: no cover
        m = self._load_pickled_file(info['object'])
        if isinstance(m, torch.nn.Module):
            m.eval()
        return m

    def _load_pickled_file(self, file_id):
        raise NotImplementedError

    def _process_documents_with_watcher(self, identifier, ids=None, verbose=False,
                                        max_chunk_size=5000, model=None, recompute=False):
        import sys
        sys.path.insert(0, os.getcwd())

        watcher_info = self.get_object_info(identifier, 'watcher')
        query_params = watcher_info['query_params']
        if ids is None:
            ids = self._get_ids_from_query(*query_params)
        if max_chunk_size is not None:
            for it, i in enumerate(range(0, len(ids), max_chunk_size)):
                print('computing chunk '
                      f'({it + 1}/{math.ceil(len(ids) / max_chunk_size)})')
                self._process_documents_with_watcher(
                    identifier,
                    ids[i: i + max_chunk_size],
                    verbose=verbose,
                    max_chunk_size=None,
                    model=model,
                    recompute=recompute,
                )
            return

        model_info = self.get_object_info(watcher_info['model'], 'model')
        outputs = self._compute_model_outputs(model_info,
                                              ids,
                                              *watcher_info['query_params'],
                                              key=watcher_info['key'],
                                              features=watcher_info.get('features', {}),
                                              model=model,
                                              loader_kwargs=watcher_info.get('loader_kwargs'),
                                              verbose=verbose)

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

    def _process_documents(self, *query_params, ids=None, verbose=False):
        job_ids = defaultdict(lambda: [])
        download_id = self._submit_download_content(*query_params, ids=ids)
        job_ids['download'].append(download_id)
        if not self.list_watchers():
            return job_ids
        G = self._create_plan()
        current = [('watcher', watcher_identifier) for watcher_identifier in self.list_watchers()
                   if not list(G.predecessors(('watcher', watcher_identifier)))]
        iteration = 0
        while current:
            for (variety, identifier) in current:
                job_ids.update(self._process_single_item(variety, identifier, iteration, job_ids,
                                                         download_id, verbose=verbose))
            current = sum([list(G.successors((variety, identifier)))
                           for (variety, identifier) in current], [])
            iteration += 1
        return job_ids

    def _process_single_item(self, variety, identifier, iteration, job_ids, download_id, ids=None,
                             verbose=True):
        if variety == 'watcher':
            watcher_info = self.get_object_info(identifier, 'watcher')
            if iteration == 0:
                dependencies = [download_id]
            else:
                model_dependencies = \
                    self._get_dependencies_for_watcher(identifier)
                dependencies = sum([
                    job_ids[('watchers', dep)]
                    for dep in model_dependencies
                ], [])
            process_id = \
                self._submit_process_documents_with_watcher(identifier, dependencies,
                                                            verbose=verbose, ids=ids)
            job_ids[(variety, identifier)].append(process_id)
            if watcher_info.get('download', False):  # pragma: no cover
                download_id = \
                    self._submit_download_content(*watcher_info['query_params'], ids=ids,
                                                  dependencies=(process_id,))
                job_ids[(variety, identifier)].append(download_id)
        elif variety == 'neighbourhoods':
            watcher_identifier = self._get_watcher_for_neighbourhood(identifier)
            dependencies = job_ids[('watchers', watcher_identifier)]
            process_id = self._submit_compute_neighbourhood(identifier, dependencies)
            job_ids[(variety, identifier)].append(process_id)
        return job_ids

    def refresh_watcher(self, identifier, dependencies=()):
        return self._submit_process_documents_with_watcher(identifier, dependencies=dependencies)

    def _replace_model(self, identifier, object):
        r = self.get_object_info(identifier, 'model')
        assert identifier in self.list_models(), f'model "{identifier}" doesn\'t exist to replace'
        if not isinstance(object, Container):
            file_id = self._create_pickled_file(object)
            self._replace_object(r['object'], file_id, 'model', identifier)
            return
        file_id = self._create_pickled_file(object._forward)
        self._replace_object(r['object'], file_id, 'model', identifier)

    def _replace_object(self, file_id, new_file_id, variety, identifier):
        raise NotImplementedError

    def save_metrics(self, identifier, variety, metrics):
        raise NotImplementedError

    def separate_query_part_from_validation_record(self, r):
        raise NotImplementedError

    def _set_content_bytes(self, r, key, bytes_):
        raise NotImplementedError

    def _submit_compute_neighbourhood(self, identifier, dependencies):
        if not self.remote:
            self._compute_neighbourhood(identifier)
        else:
            return jobs.process(
                self.name,
                '_compute_neighbourhood',
                identifier,
                dependencies=dependencies
            )

    def _submit_download_content(self, *query_params, ids=None, dependencies=()):
        if not self.remote:
            print('downloading content from retrieved urls')
            self._download_content(*query_params, ids=ids)
        else:
            return jobs.process(
                self.name,
                '_download_content',
                *query_params,
                ids=ids,
                dependencies=dependencies,
            )

    def _submit_process_documents_with_watcher(self, identifier, dependencies=(), ids=None,
                                               verbose=True):
        watcher_info = self.get_object_info(identifier, 'watcher')
        if not self.remote:
            self._process_documents_with_watcher(identifier, verbose=verbose, ids=ids)
            if watcher_info.get('download', False):  # pragma: no cover
                self._download_content(*watcher_info['query_params'])
        else:
            return jobs.process(
                self.name,
                '_process_documents_with_watcher',
                identifier,
                verbose=verbose,
                dependencies=dependencies,
            )

    def _train_imputation(self, identifier):
        import sys
        sys.path.insert(0, os.getcwd())

        if self.remote:
            return jobs.process(self.name, '_train_imputation', identifier)

        info = self.get_object_info(identifier, 'imputation')
        splitter = None
        if info.get('splitter'):
            splitter = self.splitters[info['splitter']]

        model = self.models[info['model']]
        target = self.functions[info['target']]
        objective = self.objectives[info['objective']]
        metrics = {k: self.metrics[k] for k in info['metrics']}
        keys = (info['model_key'], info['target_key'])

        training.train_tools.ImputationTrainer(
            identifier,
            models=(model, target),
            database_type=self._database_type,
            database=self.name,
            keys=keys,
            model_names=(info['model'], info['target']),
            query_params=info['query_params'],
            objective=objective,
            metrics=metrics,
            **info['trainer_kwargs'],
            save=self._replace_model,
            splitter=splitter,
        ).train()

    def _train_semantic_index(self, identifier):

        import sys
        sys.path.insert(0, os.getcwd())

        if self.remote:
            return jobs.process(self.name,
                                '_train_semantic_index',
                                identifier)

        info = self.get_object_info(identifier, 'semantic_index')
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
            identifier,
            models=models,
            keys=info['keys'],
            model_names=info['models'],
            database_type='mongodb',
            database=self.name,
            query_params=info['query_params'],
            splitter=splitter,
            objective=objective,
            save=self._replace_model,
            watch='objective',
            metrics=metrics,
            validation_sets=info.get('validation_sets', ()),
            **info.get('trainer_kwargs', {}),
        )
        t.train()

    def unset_hash_set(self, identifier):
        try:
            del self._all_hash_sets[identifier]
        except KeyError:
            pass

    def _update_neighbourhood(self, ids, similar_ids, identifier, *query_params):
        raise NotImplementedError

    def _update_object_info(self, identifier, variety, key, value):
        raise NotImplementedError

    def validate_semantic_index(self, identifier, validation_sets, metrics):
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
        try:
            status = 'pending'
            n_lines = 0
            while status in {'pending', 'running'}:
                r = self._get_job_info(identifier)
                status = r['status']
                if status == 'running':
                    if len(r['stdout']) > n_lines:
                        print(''.join(r['stdout'][n_lines:]), end='')
                    n_lines = len(r['stdout'])
                    time.sleep(0.2)
                else:
                    time.sleep(0.2)
            r = self._get_job_info(identifier)
            if status == 'success':
                if len(r['stdout']) > n_lines:
                    print(''.join(r['stdout'][n_lines:]), end='')
            elif status == 'failed': # pragma: no cover
                print(r['msg'])
        except KeyboardInterrupt: # pragma: no cover
            return

    def _write_watcher_outputs(self, info, outputs, ids):
        raise NotImplementedError

