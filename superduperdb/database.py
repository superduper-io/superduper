import io
import math
import multiprocessing
import pickle
import time
from collections import defaultdict

import click
import networkx
import torch

from superduperdb import cf, training
from superduperdb.getters import jobs
from superduperdb.getters.jobs import stop_job
from superduperdb.lookup import hashes
from superduperdb.training.validation import validate_representations
from superduperdb.types.utils import convert_from_bytes_to_types
from superduperdb.utils import gather_urls, CallableWithSecret, to_device, set_device, device_of
from superduperdb.getters import client as our_client
from superduperdb.models.utils import BasicDataset, create_container, Container, apply_model
from superduperdb.utils import ArgumentDefaultDict, progressbar, unpack_batch, Downloader


class BaseDatabase:
    """
    Base database connector for SuperDuperDB - all database types should subclass this
    type.
    """
    def __init__(self):

        self.agents = ArgumentDefaultDict(lambda x: self._load_object(x, 'agent'))
        self.functions = ArgumentDefaultDict(lambda x: self._load_object(x, 'function'))
        self.measures = ArgumentDefaultDict(lambda x: self._load_object(x, 'measure'))
        self.metrics = ArgumentDefaultDict(lambda x: self._load_object(x, 'metric'))
        self.models = ArgumentDefaultDict(lambda x: self._load_model(x))
        self.objectives = ArgumentDefaultDict(lambda x: self._load_object(x, 'objective'))
        self.preprocessors = ArgumentDefaultDict(lambda x: self._load_object(x, 'preprocessor'))
        self.postprocessors = ArgumentDefaultDict(lambda x: self._load_object(x, 'postprocessor'))
        self.splitters = ArgumentDefaultDict(lambda x: self._load_object(x, 'splitter'))
        self.trainers = ArgumentDefaultDict(lambda x: self._load_object(x, 'trainer'))
        self.types = ArgumentDefaultDict(lambda x: self._load_object(x, 'type'))

        self.remote = cf.get('remote', False)
        self._type_lookup = None

        self._hash_set = None
        self._all_hash_sets = ArgumentDefaultDict(self._load_hashes)

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

    def _apply_agent(self, agent, query_params, like=None):
        if self.remote and isinstance(agent, str):
            return our_client._apply_agent(self._database_type,
                                           self.name,
                                           agent,
                                           query_params=query_params,
                                           like=like)
        docs = list(self.execute_query(*query_params, like=like))
        if isinstance(agent, str):
            agent = self.agents[agent]
        return agent(docs)

    def apply_model(self, model, input_, **kwargs):
        """
        Apply model to input.

        :param model: PyTorch model or ``str`` referring to an uploaded model (see ``self.list_models``)
        :param input_: input_ to be passed to the model. Must be possible to encode with registered types (``self.list_types``)
        :param kwargs: key-values (see ``superduperdb.models.utils.apply_model``)
        """
        if self.remote and isinstance(model, str):
            return our_client.apply_model(self._database_type, self.name, model, input_, **kwargs)
        if isinstance(model, str):
            model = self.models[model]
        with torch.no_grad():
            return apply_model(model, input_, **kwargs)

    def _build_processing_graph(self):
        ...

    def cancel_job(self, job_id):
        stop_job(self._database_type, self.name, job_id)

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
                batch = to_device(batch, device_of(model))
                with torch.no_grad():
                    output = model.forward(batch)
                if has_post:
                    unpacked = unpack_batch(output)
                    unpacked = to_device(unpacked, 'cpu')
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
        """
        Convert the non-Jsonable python objects in a nested dictionary into ``bytes``

        :param r: dictionary potentially containing non-Jsonable content
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

    def _convert_id_to_str(self, id_):
        raise NotImplementedError

    def _convert_ids_to_strs(self, ids):
        return [self._convert_id_to_str(id_) for id_ in ids]

    def _convert_str_to_id(self, id_):
        raise NotImplementedError

    def _convert_strs_to_ids(self, strs):
        return [self._convert_str_to_id(id_) for id_ in strs]

    def create_agent(self, identifier, object, **kwargs):
        """
        Create agent - callable function, which is not a PyTorch model.

        :param identifier: identifier
        :param object: Python object
        """
        return self._create_object(identifier, object, 'agent', **kwargs)

    def create_function(self, identifier, object, **kwargs):
        """
        Create function - callable function, which is not a PyTorch model.

        :param identifier: identifier
        :param object: Python object
        """
        return self._create_object(identifier, object, 'function', **kwargs)

    def _create_imputation(self, identifier, model, model_key, target, target_key, query_params,
                           objective=None, metrics=None, validation_sets=(),
                           splitter=None, loader_kwargs=None, trainer_kwargs=None):
        return self._create_learning_task(
            identifier,
            'ImputationTrainer',
            'imputation',
            models=[model, target],
            keys=[model_key, target_key],
            query_params=query_params,
            objective=objective,
            metrics=metrics,
            validation_sets=validation_sets,
            splitter=splitter,
            loader_kwargs=loader_kwargs,
            trainer_kwargs=trainer_kwargs,
            keys_to_watch=[model_key],
        )

    def _create_job_record(self, *args, **kwargs):
        raise NotImplementedError

    def _create_learning_task(self, identifier, trainer, task_type, models, keys, query_params,
                              verbose=False,
                              validation_sets=(), metrics=(), objective=None, keys_to_watch=(),
                              loader_kwargs=None, splitter=None, flags=None, trainer_kwargs=None):

        assert trainer in [*self.list_trainers(), 'SemanticIndexTrainer', 'ImputationTrainer']
        assert identifier not in self.list_learning_tasks()
        for k in keys_to_watch:
            assert f'{identifier}/{k}' not in self.list_watchers()
        flags = flags or {}
        trainer_kwargs = trainer_kwargs or {}

        self._create_object_entry({
            'variety': 'learning_task',
            'identifier': identifier,
            'query_params': query_params,
            'models': models,
            'keys': keys,
            'metrics': metrics,
            'objective': objective,
            'splitter': splitter,
            'validation_sets': validation_sets,
            'keys_to_watch': keys_to_watch,
            'trainer_kwargs': trainer_kwargs,
            'trainer': trainer,
            'task_type': task_type,
            **flags,
        })

        model_lookup = dict(zip(keys_to_watch, models))
        jobs = []
        if objective is not None:
            try:
                jobs.append(self._train(identifier, task_type))
            except KeyboardInterrupt:
                print('training aborted...')

        for k in keys_to_watch:
            jobs.append(self._create_watcher(
                f'{model_lookup[k]}/{k}',
                model_lookup[k],
                query_params,
                key=k,
                features=trainer_kwargs.get('features', {}),
                loader_kwargs=loader_kwargs,
                dependencies=jobs,
                verbose=verbose
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

    def create_model(self, identifier, object=None, preprocessor=None, postprocessor=None,
                     type=None, **kwargs):
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
            with set_device(object, 'cpu'):
                file_id = self._create_serialized_file(object, **kwargs)

        self._create_object_entry({
            'variety': 'model',
            'identifier': identifier,
            'object': file_id,
            'type': type,
            'preprocessor': preprocessor,
            'postprocessor': postprocessor,
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
        if not self.remote:
            self._compute_neighbourhood(identifier)
        else:
            return jobs.process(
                self._database_type,
                self.name,
                '_compute_neighbourhood',
                identifier,
            )

    def _create_object_entry(self, info):
        raise NotImplementedError

    def _create_object(self, identifier, object, variety, serializer='pickle', serializer_kwargs=None):
        serializer_kwargs = serializer_kwargs or {}
        assert identifier not in self._list_objects(variety)
        secrets = {}
        if isinstance(object, CallableWithSecret):
            secrets = {'secrets': object.secrets}
            object.secrets = None
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

    def create_objective(self, identifier, object, **kwargs):
        """
        Create differentiable objective function, called by ``self.create_learning_task``,
        to smoothly measure performance of learning on training set for back-propagation.

        :param identifier: identifier
        :param object: Python object
        """
        return self._create_object(identifier, object, 'objective', **kwargs)

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

    def create_postprocessor(self, identifier, object, **kwargs):
        return self._create_object(identifier, object, 'postprocessor', **kwargs)

    def create_preprocessor(self, identifier, object, **kwargs):
        return self._create_object(identifier, object, 'preprocessor', **kwargs)

    def _create_semantic_index(self, identifier, models, keys, measure, query_params,
                               validation_sets=(), metrics=(), objective=None, index_type='vanilla',
                               verbose=False,
                               index_kwargs=None, splitter=None, loader_kwargs=None,
                               trainer_kwargs=None):

        for vs in validation_sets:
            tmp_query_params = self.get_query_params_for_validation_set(vs)
            self._create_semantic_index(
                identifier + f'/{vs}',
                models,
                keys,
                measure,
                tmp_query_params,
                loader_kwargs=loader_kwargs,
            )

        return self._create_learning_task(
            identifier,
            trainer='SemanticIndexTrainer',
            task_type='semantic_index',
            models=models,
            keys=keys,
            query_params=query_params,
            validation_sets=validation_sets,
            metrics=metrics,
            objective=objective,
            flags={
                'index_type': index_type,
                'index_kwargs': index_kwargs,
                'measure': measure,
            },
            splitter=splitter,
            loader_kwargs=loader_kwargs,
            trainer_kwargs=trainer_kwargs,
            keys_to_watch=[keys[0]],
            verbose=verbose,
        )

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

    def create_splitter(self, identifier, object, **kwargs):
        return self._create_object(identifier, object, 'splitter', **kwargs)

    def create_trainer(self, identifier, object, **kwargs):
        """
        Create trainer class, called by ``self.create_learning_task``.

        :param identifier: identifier
        :param object: Python object
        """
        return self._create_object(identifier, object, 'trainer', **kwargs)

    def create_type(self, identifier, object, **kwargs):
        """
        Create datatype, in order to serialize python object data in the database.

        :param identifier: identifier
        :param object: Python object
        """
        return self._create_object(identifier, object, 'type', **kwargs)

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

            return self._submit_process_documents_with_watcher(
                identifier,
                ids=ids,
                verbose=verbose,
                dependencies=dependencies,
            )

    def delete_agent(self, identifier, force=False):
        """
        Delete agent.

        :param identifier: identifier of agent
        :param force: toggle to ``True`` to skip confirmation step
        """
        return self._delete_object(identifier, 'agent', force=force)

    def delete_function(self, identifier, force=False):
        """
        Delete function.

        :param identifier: identifier of function
        :param force: toggle to ``True`` to skip confirmation step
        """
        return self._delete_object(identifier, 'function', force=force)

    def delete_imputation(self, identifier, force=False):
        """
        Delete imputation

        :param identifier: Identifier of imputation
        :param force: Toggle to ``True`` to skip confirmation
        """
        return self.delete_learning_task(identifier, force=force)

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

        model_lookup = dict(zip(info['models'], info['keys']))
        for k in info['keys_to_watch']:
            self._delete_object_info(f'{model_lookup[k]}/{k}', 'watcher')
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
            print('aborting') # pragma: no cover

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

    def delete_objective(self, identifier, force=False):
        """
        Delete objective.

        :param identifier: name of objective
        :param force: toggle to ``True`` to skip confirmation step
        """
        return self._delete_object(identifier, 'objective', force=force)

    def delete_postprocessor(self, identifier, force=False):
        """
        Delete postprocessor.

        :param identifier: name of postprocessor
        :param force: toggle to ``True`` to skip confirmation step
        """
        return self._delete_object(identifier, 'postprocessor', force=force)

    def delete_preprocessor(self, identifier, force=False):
        """
        Delete preprocessor.

        :param identifier: name of preprocessor
        :param force: toggle to ``True`` to skip confirmation step
        """
        return self._delete_object(identifier, 'preprocessor', force=force)

    def delete_semantic_index(self, identifier, force=False):
        """
        Delete semantic-index.

        :param name: name of semantic-index
        :param force: toggle to ``True`` to skip confirmation step
        """
        info = self.get_object_info(identifier, 'learning_task')
        watcher_info = self.get_object_info(f'{identifier}/{info["keys"][0]}', 'watcher')
        if info is None:  # pragma: no cover
            return
        do_delete = False
        if force or click.confirm(f'Are you sure you want to delete this semantic index: '
                                  f'"{identifier}"; '):
            do_delete = True

        if not do_delete:
            return

        if watcher_info:
            self.delete_watcher(f'{identifier}/{info["keys"][0]}', force=True)
        self._delete_object_info(identifier, 'learning_task')

    def delete_splitter(self, identifier, force=False):
        """
        Delete splitter.

        :param identifier: identifier of splitter
        :param force: toggle to ``True`` to skip confirmation step
        """
        return self._delete_object(identifier, 'splitter', force=force)

    def delete_trainer(self, identifier, force=False):
        """
        Delete trainer.

        :param identifier: identifier of trainer
        :param force: toggle to ``True`` to skip confirmation step
        """
        return self._delete_object(identifier, 'trainer', force=force)

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

    def _download_content(self, table, query_params, ids=None, documents=None, timeout=None,
                          raises=True, n_download_workers=None, headers=None):
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
        dependencies = []
        if watcher_features:
            for key, model in watcher_features.items():
                dependencies.append(f'{model}/{key}')
        return dependencies

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

    def _get_object_info(self, identifier, variety, **kwargs):
        raise NotImplementedError

    def _get_object_info_where(self, variety, **kwargs):
        raise NotImplementedError

    def get_object_info(self, identifier, variety, decode_types=False, **kwargs):
        r = self._get_object_info(identifier, variety, **kwargs)
        if decode_types:
            r = convert_from_bytes_to_types(r, converters=self.types)
        return r

    def get_object_info_where(self, variety, **kwargs):
        return self._get_object_info_where(variety, **kwargs)

    def get_query_params_for_validation_set(self, validation_set):
        raise NotImplementedError

    def _get_watcher_for_neighbourhood(self, identifier):
        return self.get_object_info(identifier, 'neighbourhood')['watcher_identifier']

    def _insert_validation_data(self, tmp, identifier):
        raise NotImplementedError

    def list_agents(self):
        """
        List agents.
        """
        return self._list_objects('agent')

    def list_functions(self):
        """
        List functions.
        """
        return self._list_objects('function')

    def list_imputations(self):
        """
        List imputations.
        """
        return self._list_objects('learning_task', task_type='imputation')

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

    def list_objectives(self):
        """
        List objectives.
        """
        return self._list_objects('objective')

    def list_splitters(self):
        """
        List splitters.
        """
        return self._list_objects('splitter')

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

    def list_semantic_indexes(self):
        """
        List semantic indexes.
        """
        return self._list_objects('learning_task', task_type='semantic_index')

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

    def _load_hashes(self, identifier):
        info = self.get_object_info(identifier, 'learning_task', task_type='semantic_index')
        index_type = info.get('index_type', 'vanilla')
        index_kwargs = info.get('index_kwargs', {}) or {}
        key_to_watch = info['keys_to_watch'][0]
        watcher_info = self.get_object_info(f'{identifier}/{key_to_watch}', 'watcher')
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

        if index_type == 'vanilla':
            return hashes.HashSet(torch.stack(loaded), ids, measure=measure)
        elif index_type == 'faiss':
            return hashes.FaissHashSet(torch.stack(loaded), ids, **index_kwargs)
        else:
            raise NotImplementedError

    def _load_model(self, identifier):
        info = self.get_object_info(identifier, 'model')
        if info is None:
            raise Exception(f'No such object of type "model", "{identifier}" has been registered.') # pragma: no cover
        info = dict(info)
        model = self._load_pickled_file(info['object'])
        model.eval()
        if torch.cuda.is_available():
            model.to('cuda')

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
        if 'serializer' not in info:
            info['serializer'] = 'pickle'
        if 'serializer_kwargs' not in info:
            info['serializer_kwargs'] = {}
        m = self._load_pickled_file(info['object'], serializer=info['serializer'])
        if isinstance(m, CallableWithSecret) and 'secrets' in info:
            m.secrets = info['secrets']
        if isinstance(m, torch.nn.Module):
            m.eval()
        return m

    def _load_pickled_file(self, file_id, serializer='pickle'):
        bytes_ = self._load_blob_of_bytes(file_id)
        f = io.BytesIO(bytes_)
        if serializer == 'pickle':
            return pickle.load(f)
        elif serializer == 'dill':
            import dill
            return dill.load(f)
        raise NotImplementedError

    def _process_documents_with_watcher(self, identifier, ids=None, verbose=False,
                                        max_chunk_size=5000, model=None, recompute=False):
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

    def _process_documents(self, query_params, ids=None, verbose=False, dependencies=()):
        job_ids = defaultdict(lambda: [])
        job_ids.update(dependencies)
        dependencies = sum([list(v) for k, v in dependencies.items()], [])
        if not self.list_watchers():
            return job_ids
        G = self._create_plan()
        current = [('watcher', watcher_identifier) for watcher_identifier in self.list_watchers()
                   if not list(G.predecessors(('watcher', watcher_identifier)))]
        iteration = 0
        while current:
            for (variety, identifier) in current:
                job_ids.update(self._process_single_item(variety, identifier, iteration, job_ids,
                                                         dependencies, ids=ids, verbose=verbose))
            current = sum([list(G.successors((variety, identifier)))
                           for (variety, identifier) in current], [])
            iteration += 1
        return job_ids

    def _process_single_item(self, variety, identifier, iteration, job_ids, dependencies, ids=None,
                             verbose=True):
        if variety == 'watcher':
            watcher_info = self.get_object_info(identifier, 'watcher')
            if iteration == 0:
                pass
            else:
                model_dependencies = \
                    self._get_dependencies_for_watcher(identifier)
                dependencies = sum([
                    job_ids[('watcher', dep)]
                    for dep in model_dependencies
                ], [])
            process_id = \
                self._submit_process_documents_with_watcher(identifier, dependencies,
                                                            verbose=verbose, ids=ids)
            job_ids[(variety, identifier)].append(process_id)
            if watcher_info.get('_download', False):  # pragma: no cover
                download_id = \
                    self._submit_download_content(*watcher_info['query_params'], ids=ids,
                                                  dependencies=(process_id,))
                job_ids[(variety, identifier)].append(download_id)
        elif variety == 'neighbourhoods':
            watcher_identifier = self._get_watcher_for_neighbourhood(identifier)
            dependencies = job_ids[('watcher', watcher_identifier)]
            process_id = self._submit_compute_neighbourhood(identifier, dependencies)
            job_ids[(variety, identifier)].append(process_id)
        return job_ids

    def refresh_watcher(self, identifier, dependencies=()):
        """
        Recompute outputs of watcher

        :param identifier: identifier of watcher
        :param dependencies: job-ids on which computation should depend
        """
        return self._submit_process_documents_with_watcher(identifier, dependencies=dependencies)

    def _replace_model(self, identifier, object):
        info = self.get_object_info(identifier, 'model')
        if 'serializer' not in info:
            info['serializer'] = 'pickle'
        if 'serializer_kwargs' not in info:
            info['serializer_kwargs'] = {}
        assert identifier in self.list_models(), f'model "{identifier}" doesn\'t exist to replace'
        if not isinstance(object, Container):
            with set_device(object, 'cpu'):
                file_id = self._create_serialized_file(object, serializer=info['serializer'],
                                                       serializer_kwargs=info['serializer_kwargs'])
            self._replace_object(info['object'], file_id, 'model', identifier)
            return

        with set_device(object, 'cpu'):
            file_id = self._create_serialized_file(object._forward)

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

    def _submit_compute_neighbourhood(self, identifier, dependencies):
        if not self.remote:
            self._compute_neighbourhood(identifier)
        else:
            return jobs.process(
                self._database_type,
                self.name,
                '_compute_neighbourhood',
                identifier,
                dependencies=dependencies
            )

    def _submit_download_content(self, query_params, ids=None, dependencies=()):
        if not self.remote:
            print('downloading content from retrieved urls')
            ids = self._convert_strs_to_ids(ids)
            self._download_content(query_params, ids=ids)
        else:
            ids = self._convert_ids_to_strs(ids)
            return jobs.process(
                self._database_type,
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
            if ids is not None:
                ids = self._convert_strs_to_ids(ids)
            self._process_documents_with_watcher(identifier, verbose=verbose, ids=ids)
            if watcher_info.get('_download', False):  # pragma: no cover
                self._download_content(*watcher_info['query_params'])
        else:
            ids = self._convert_ids_to_strs(ids)
            return jobs.process(
                self._database_type,
                self.name,
                '_process_documents_with_watcher',
                identifier,
                ids=ids,
                verbose=verbose,
                dependencies=dependencies,
            )

    def _get_trainer(self, identifier, train_type):
        info = self.get_object_info(identifier, 'learning_task')
        splitter = None
        if info.get('splitter'):
            splitter = self.splitters[info['splitter']]

        if info['trainer'] == 'SemanticIndexTrainer':
            trainer_cls = training.trainer.SemanticIndexTrainer
        elif info['trainer'] == 'ImputationTrainer':
            trainer_cls = training.trainer.ImputationTrainer
        else:
            trainer_cls = self._load_object(info['trainer'], 'trainer')

        models = []
        for m in info['models']:
            try:
                models.append(self.models[m])
            except Exception as e:
                if 'No such object of type' in str(e):
                    models.append(self.functions[m])
                else:
                    raise e
        objective = self.objectives[info['objective']]
        metrics = {k: self.metrics[k] for k in info['metrics']}

        trainer = trainer_cls(
            identifier,
            models=models,
            database_type=self._database_type,
            database=self.name,
            keys=info['keys'],
            model_names=info['models'],
            query_params=info['query_params'],
            objective=objective,
            metrics=metrics,
            save=self._replace_model,
            splitter=splitter,
            validation_sets=info.get('validation_sets', ()),
            **info.get('trainer_kwargs', {}),
        )
        return trainer

    def _train(self, identifier, train_type):
        if self.remote:
            return jobs.process(self._database_type, self.name, '_train', identifier, train_type)

        info = self.get_object_info(identifier, 'learning_task')
        splitter = None
        if info.get('splitter'):
            splitter = self.splitters[info['splitter']]

        if info['trainer'] == 'SemanticIndexTrainer':
            trainer_cls = training.trainer.SemanticIndexTrainer
        elif info['trainer'] == 'ImputationTrainer':
            trainer_cls = training.trainer.ImputationTrainer
        else:
            trainer_cls = self._load_object(info['trainer'], 'trainer')

        models = []
        for m in info['models']:
            try:
                models.append(self.models[m])
            except Exception as e:
                if 'No such object of type' in str(e):
                    models.append(self.functions[m])
                else:
                    raise e
        objective = self.objectives[info['objective']]
        metrics = {k: self.metrics[k] for k in info['metrics']}

        trainer = trainer_cls(
            identifier,
            models=models,
            database_type=self._database_type,
            database=self.name,
            keys=info['keys'],
            model_names=info['models'],
            query_params=info['query_params'],
            objective=objective,
            metrics=metrics,
            save=self._replace_model,
            splitter=splitter,
            validation_sets=info.get('validation_sets', ()),
            **info.get('trainer_kwargs', {}),
        )
        trainer.train()

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

    def _write_watcher_outputs(self, info, outputs, ids):
        raise NotImplementedError

