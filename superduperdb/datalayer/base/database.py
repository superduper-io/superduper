import warnings
from collections import defaultdict
import math
import random
from typing import Any, Union, Optional, Dict

import click
import networkx
from bson import ObjectId

from superduperdb import CFG, misc
from superduperdb.cluster.client_decorators import model_server, vector_search
from superduperdb.cluster.annotations import Convertible, Tuple, List
from superduperdb.cluster.job_submission import work
from superduperdb.cluster.task_workflow import TaskWorkflow
from superduperdb.core.base import Component, strip
from superduperdb.core.exceptions import ComponentInUseError, ComponentInUseWarning
from superduperdb.core.learning_task import LearningTask
from superduperdb.core.model import Model
from superduperdb.core.vector_index import VectorIndex
from superduperdb.datalayer.base.artifacts import ArtifactStore
from superduperdb.datalayer.base.metadata import MetaDataStore
from superduperdb.datalayer.base.query import Insert, Select, Delete, Update
from superduperdb.fetchers.downloads import gather_uris
from superduperdb.misc.special_dicts import ArgumentDefaultDict
from superduperdb.fetchers.downloads import Downloader
from superduperdb.misc import progress
from superduperdb.misc.logger import logging


class BaseDatabase:
    """
    Base database connector for SuperDuperDB - all database types should subclass this
    type.
    """

    select_cls = Select
    variety_to_cache_mapping = {
        'model': 'models',
        'metric': 'metrics',
        'type': 'types',
        'vector_index': 'vector_indices',
    }

    def __init__(self, metadata: MetaDataStore, artifact_store: ArtifactStore):
        self.metrics = ArgumentDefaultDict(lambda x: self.load_component(x, 'metric'))
        self.models = ArgumentDefaultDict(lambda x: self.load_component(x, 'model'))
        self.types = ArgumentDefaultDict(lambda x: self.load_component(x, 'type'))
        self.vector_indices = ArgumentDefaultDict(
            lambda x: self.load_component(x, 'vector_index')
        )

        self.remote = CFG.remote
        self._type_lookup = None

        self.metadata = metadata
        self.artifact_store = artifact_store

    def _reload_type_lookup(self):
        self._type_lookup = {}
        for t in self.list_components('type'):
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
        :param input_: input_ to be passed to the model.
                       Must be possible to encode with registered types
        :param kwargs: key-values (see ``superduperdb.models.utils.predict``)
        """
        if isinstance(model, str):
            model = self.models[model]
        return model.predict(input_, **kwargs)

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
        if not self.list_components('watcher'):
            return G

        for identifier in self.list_components('watcher'):
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

        for identifier in self.list_components('watcher'):
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

        return G

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

    def _create_job_record(self, *args, **kwargs):  # TODO - move to metadata
        raise NotImplementedError

    def create_component(
        self,
        object: Component,
        serializer: str = 'pickle',
        serializer_kwargs: Optional[Dict] = None,
        parent: Optional[str] = None,
    ):
        existing_versions = self.list_component_versions(
            object.variety, object.identifier
        )
        if isinstance(object.version, int) and object.version in existing_versions:
            logging.warn(f'{object.unique_id} already exists - doing nothing')
            return
        version = existing_versions[-1] + 1 if existing_versions else 0
        object.version = version

        for c in object.child_components:
            logging.info(f'Checking upstream-component {c.variety}/{c.identifier}')
            self.create_component(
                c,
                serializer=serializer,
                serializer_kwargs=serializer_kwargs,
                parent=object.unique_id,
            )

        for p in object.child_references:
            if p.version is None:
                p.version = self.metadata.get_latest_version(p.variety, p.identifier)

        print('Stripping sub-components to references')
        strip(object)

        serializer_kwargs = serializer_kwargs or {}
        file_id, sha1 = self.artifact_store.create_artifact(
            object,
            serializer=serializer,
            serializer_kwargs=serializer_kwargs,
        )
        self.metadata.create_component(
            {
                **object.asdict(),
                'object': file_id,
                'variety': object.variety,
                'version': version,
                'sha1': sha1,
            }
        )
        if parent is not None:
            self.metadata.create_parent_child(parent, object.unique_id)
        logging.info(f'Created {object.unique_id}')

        object.repopulate(self)
        return object.schedule_jobs(self)

    def _create_plan(self):
        G = networkx.DiGraph()
        for identifier in self.metadata.list_components('watcher'):
            G.add_node(('watcher', identifier))
        for identifier in self.metadata.list_components('watcher'):
            deps = self._get_dependencies_for_watcher(identifier)
            for dep in deps:
                G.add_edge(('watcher', dep), ('watcher', identifier))
        assert networkx.is_directed_acyclic_graph(G)
        return G

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

    def delete_component_version(
        self,
        variety: str,
        identifier: str,
        version: int,
        force: bool = False,
    ):
        unique_id = Component.make_unique_id(variety, identifier, version)
        if self.metadata.component_version_has_parents(variety, identifier, version):
            parents = self.metadata.get_component_version_parents(
                variety, identifier, version
            )
            raise Exception(f'{unique_id} is involved in other components: {parents}')

        if force or click.confirm(
            f'You are about to delete {unique_id}, are you sure?',
            default=False,
        ):
            info = self.metadata.get_component(variety, identifier, version=version)
            if variety in self.variety_to_cache_mapping:
                try:
                    del getattr(self, self.variety_to_cache_mapping[variety])[
                        identifier
                    ]
                except KeyError:
                    pass
            self.artifact_store.delete_artifact(info['object'])
            self.metadata.delete_component_version(variety, identifier, version=version)

    def delete_component(self, variety, identifier, force=False):
        versions = self.metadata.list_component_versions(variety, identifier)
        versions_in_use = []
        for v in versions:
            if self.metadata.component_version_has_parents(variety, identifier, v):
                versions_in_use.append(v)

        if versions_in_use:
            component_versions_in_use = []
            for v in versions_in_use:
                unique_id = Component.make_unique_id(variety, identifier, v)
                component_versions_in_use.append(
                    f"{unique_id} -> "
                    f"{self.metadata.get_component_version_parents(unique_id)}",
                )
            if not force:
                raise ComponentInUseError(
                    f'Component versions: {component_versions_in_use} are in use'
                )
            else:
                warnings.warn(
                    ComponentInUseWarning(
                        f'Component versions: {component_versions_in_use}'
                        ', marking as hidden'
                    )
                )

        if force or click.confirm(
            f'You are about to delete {variety}/{identifier}, are you sure?',
            default=False,
        ):
            for v in sorted(list(set(versions) - set(versions_in_use))):
                self.delete_component_version(variety, identifier, v, force=True)

            for v in sorted(versions_in_use):
                self.metadata.hide_component_version(variety, identifier, v)
        else:
            print('aborting.')

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
                n_download_workers = self.metadata.get_metadata(
                    key='n_download_workers'
                )
            except TypeError:
                n_download_workers = 0

        if headers is None:
            try:
                headers = self.metadata.get_metadata(key='headers')
            except TypeError:
                headers = 0

        if timeout is None:
            try:
                timeout = self.metadata.get_metadata(key='download_timeout')
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
        info = self.metadata.get_component('watcher', identifier)
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

    def get_object_info(self, identifier, variety):
        return self.metadata.get_component(variety, identifier)

    def get_query_for_validation_set(self, validation_set):
        raise NotImplementedError

    def _get_watcher_for_learning_task(self, learning_task):
        info = self.metadata.get_component('learning_task', learning_task)
        key_to_watch = info['keys_to_watch'][0]
        model_identifier = next(
            m for i, m in enumerate(info['models']) if info['keys'][i] == key_to_watch
        )
        return f'[{learning_task}]:{model_identifier}/{key_to_watch}'

    def insert(self, insert: Insert, refresh=True, verbose=True):
        for item in insert.documents:
            r = random.random()
            try:
                valid_probability = self.metadata.get_metadata(key='valid_probability')
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
        return self.metadata.list_jobs()

    def list_components(self, variety):
        return self.metadata.list_components(variety)

    def list_component_versions(self, variety: str, identifier: str):
        return sorted(self.metadata.list_component_versions(variety, identifier))

    def list_validation_sets(self):
        """
        List validation sets.
        """
        return self.metadata.list_components(variety='validation_set')

    def load_component(
        self,
        identifier: str,
        variety: str,
        version: Optional[int] = None,
        repopulate: bool = True,
        allow_hidden: bool = False,
    ):
        info = self.metadata.get_component(
            variety, identifier, version=version, allow_hidden=allow_hidden
        )
        if info is None:
            raise Exception(
                f'No such object of type "{variety}", '
                f'"{identifier}" has been registered.'
            )
        if 'serializer' not in info:
            info['serializer'] = 'pickle'
        if 'serializer_kwargs' not in info:
            info['serializer_kwargs'] = {}
        m = self.artifact_store.load_artifact(
            info['object'], serializer=info['serializer']
        )
        if repopulate:
            m.repopulate(self)
        if cm := self.variety_to_cache_mapping.get(variety):
            getattr(self, cm)[m.identifier] = m
        return m

    @work
    def apply_watcher(  # noqa: F811
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
            watcher_info = self.metadata.get_component('watcher', identifier)
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

        model_info = self.metadata.get_component('model', watcher_info['model'])
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

    def _replace_model(self, identifier: str, object: Model):
        info = self.metadata.get_component('model', identifier, version=object.version)
        if 'serializer' not in info:
            info['serializer'] = 'pickle'
        if 'serializer_kwargs' not in info:
            info['serializer_kwargs'] = {}
        assert identifier in self.metadata.list_components(
            'model'
        ), f'model "{identifier}" doesn\'t exist to replace'
        assert object.version in self.metadata.list_component_versions(
            'model', identifier
        )

        file_id = self.artifact_store.create_artifact(
            object,
            serializer=info['serializer'],
            serializer_kwargs=info['serializer_kwargs'],
        )
        self.artifact_store.delete_artifact(info['object'])
        self.metadata.update_object(identifier, 'model', 'object', file_id)

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
        outputs=None,
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
                    outputs=outputs,
                )
            else:
                return self._select_matches_then_similar(
                    like,
                    vector_index,
                    select,
                    raw=raw,
                    n=n,
                    features=features,
                    outputs=outputs,
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
        outputs=None,
    ):
        if not select.is_trivial:
            id_cursor = self._get_raw_cursor(select.select_only_id)
            ids = [x['_id'] for x in id_cursor]
            similar_ids, scores = self.select_nearest(
                like,
                ids=ids,
                n=n,
                vector_index=vector_index,
                outputs=outputs,
            )
        else:
            similar_ids, scores = self.select_nearest(
                like, n=n, vector_index=vector_index, outputs=outputs
            )

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
        outputs=None,
    ):
        similar_ids, scores = self.select_nearest(
            like,
            n=n,
            vector_index=vector_index,
            outputs=outputs,
        )

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
        outputs: Optional[dict] = None,
        n=10,
    ) -> Tuple([List(Convertible()), Any]):
        vector_index: VectorIndex = self.vector_indices[vector_index]
        return vector_index.get_nearest(
            like, database=self, ids=ids, n=n, outputs=outputs
        )

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
            self.delete_component('learning_task', identifier, force=True)
            raise e

    def update(self, update: Update, refresh=True, verbose=True):
        if refresh and self.metadata.list_components('model'):
            ids = self._get_ids_from_select(update.select_ids)
        result = self._base_update(update.to_raw(self.types, self.type_lookup))
        if refresh and self.metadata.list_components('model'):
            task_graph = self._build_task_workflow(
                update.select, ids=ids, verbose=verbose
            )
            task_graph()
            return result, task_graph
        return result

    @work
    def validate_component(
        self,
        identifier: str,
        variety: str,
        validation_sets: List(str),
        metrics: List(str),
    ):
        """
        Evaluate quality of component, using `Component.validate`, if implemented.

        :param identifier: identifier of semantic index
        :param variety: variety of component
        :param validation_sets: validation-sets on which to validate
        :param metrics: metric functions to compute
        """
        component = self.load_component(identifier, variety)
        metrics = [self.load_component(m, 'metric') for m in metrics]
        validation_selects = [
            self.get_query_for_validation_set(vs) for vs in validation_sets
        ]
        results = component.validate(self, validation_selects, metrics)
        for vs, res in zip(validation_sets, results):
            for m in res:
                self.metadata.update_object(
                    identifier,
                    variety,
                    f'final_metrics.{vs}.{m}',
                    res[m],
                )

    def watch_job(self, identifier):
        """
        Watch stdout/stderr of worker job.

        :param identifier: job-id
        """
        return self.metadata.watch_job(identifier)

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
