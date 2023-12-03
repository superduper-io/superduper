import dataclasses as dc
import math
import random
import typing as t
import warnings
from collections import defaultdict

import click
import networkx
import tqdm

import superduperdb as s
from superduperdb import logging
from superduperdb.backends.base.artifact import ArtifactStore
from superduperdb.backends.base.backends import vector_searcher_implementations
from superduperdb.backends.base.compute import ComputeBackend
from superduperdb.backends.base.data_backend import BaseDataBackend
from superduperdb.backends.base.metadata import MetaDataStore
from superduperdb.backends.base.query import Delete, Insert, RawQuery, Select, Update
from superduperdb.backends.ibis.query import Table
from superduperdb.backends.local.compute import LocalComputeBackend
from superduperdb.base import exceptions, serializable
from superduperdb.base.cursor import SuperDuperCursor
from superduperdb.base.document import Document
from superduperdb.base.superduper import superduper
from superduperdb.cdc.cdc import DatabaseChangeDataCapture
from superduperdb.components.component import Component
from superduperdb.components.encoder import Encodable, Encoder
from superduperdb.components.model import Model
from superduperdb.jobs.job import ComponentJob, FunctionJob, Job
from superduperdb.jobs.task_workflow import TaskWorkflow
from superduperdb.misc.colors import Colors
from superduperdb.misc.data import ibatch
from superduperdb.misc.download import Downloader, download_content, gather_uris
from superduperdb.misc.special_dicts import MongoStyleDict
from superduperdb.vector_search.base import BaseVectorSearcher, VectorItem
from superduperdb.vector_search.interface import FastVectorSearcher
from superduperdb.vector_search.update_tasks import copy_vectors, delete_vectors

DBResult = t.Any
TaskGraph = t.Any

DeleteResult = DBResult
InsertResult = t.Tuple[DBResult, t.Optional[TaskGraph]]
SelectResult = SuperDuperCursor
UpdateResult = t.Any

ExecuteQuery = t.Union[Select, Delete, Update, Insert, str]
ExecuteResult = t.Union[SelectResult, DeleteResult, UpdateResult, InsertResult]

ENDPOINTS = 'delete', 'execute', 'insert', 'like', 'select', 'update'


class Datalayer:
    """
    Base database connector for SuperDuperDB
    """

    type_id_to_cache_mapping = {
        'model': 'models',
        'metric': 'metrics',
        'encoder': 'encoders',
        'vector_index': 'vector_indices',
    }

    def __init__(
        self,
        databackend: BaseDataBackend,
        metadata: MetaDataStore,
        artifact_store: ArtifactStore,
        compute: ComputeBackend = LocalComputeBackend(),
    ):
        """
        :param databackend: object containing connection to Datastore
        :param metadata: object containing connection to Metadatastore
        :param artifact_store: object containing connection to Artifactstore
        :param compute: object containing connection to ComputeBackend
        """
        logging.info("Building Data Layer")

        self.metrics = LoadDict(self, field='metric')
        self.models = LoadDict(self, field='model')
        self.encoders = LoadDict(self, field='encoder')
        self.vector_indices = LoadDict(self, field='vector_index')
        self.fast_vector_searchers = LoadDict(
            self, callable=self.initialize_vector_searcher
        )
        self.metadata = metadata
        self.artifact_store = artifact_store
        self.databackend = databackend
        # TODO: force set config stores connection url

        self.cdc = DatabaseChangeDataCapture(self)

        self.compute = compute
        self._server_mode = False

    def rebuild(self, cfg=None):
        from superduperdb.base import build

        self.databackend = build.build_databackend(cfg.data_backend if cfg else None)
        self.compute = build.build_compute(cfg.cluster.compute if cfg else None)

        if cfg:
            self.metadata = build.build_metadata(cfg.metadata_store)
            self.artifact_store = build.build_artifact_store(cfg.artifact_store)
        else:
            self.metadata = self.databackend.build_metadata()
            self.artifact_store = self.databackend.build_artifact_store()

    @property
    def server_mode(self):
        return self._server_mode

    @server_mode.setter
    def server_mode(self, is_server: bool):
        assert isinstance(is_server, bool)
        self._server_mode = is_server

    def initialize_vector_searcher(
        self, identifier, searcher_type: t.Optional[str] = None, backfill=False
    ) -> BaseVectorSearcher:
        searcher_type = searcher_type or s.CFG.vector_search
        logging.info(f"loading of vectors of vector-index: '{identifier}'")
        vi = self.vector_indices[identifier]

        clt = vi.indexing_listener.select.table_or_collection

        if self.cdc.running:
            msg = 'CDC only supported for vector search via lance format'
            assert s.CFG.vector_search == 'lance', msg

        vector_search_cls = vector_searcher_implementations[searcher_type]
        vector_comparison = vector_search_cls(
            identifier=vi.identifier,
            dimensions=vi.dimensions,
            measure=vi.measure,
        )
        assert isinstance(clt.identifier, str), 'clt.identifier must be a string'

        if self.cdc.running:
            # In this case, loading has already happened on disk via CDC mechanism
            return vector_comparison
        if backfill or s.CFG.cluster.vector_search is None:
            self.backfill_vector_search(vi, vector_comparison)

        return FastVectorSearcher(self, vector_comparison, vi.identifier)

    def backfill_vector_search(self, vi, searcher):
        if vi.indexing_listener.select is None:
            raise ValueError('.select must be set')

        key = vi.indexing_listener.key
        if key.startswith('_outputs.'):
            key = key.split('.')[1]

        model_id = vi.indexing_listener.model.identifier
        model_version = vi.indexing_listener.model.version

        query = vi.indexing_listener.select.outputs(
            **{key: f'{model_id}/{model_version}'}
        )

        logging.info(str(query))

        progress = tqdm.tqdm(desc='Loading vectors into vector-table...')
        for record_batch in ibatch(
            self.execute(query),
            s.CFG.cluster.backfill_batch_size,
        ):
            items = []
            for record in record_batch:
                key = vi.indexing_listener.key
                if key.startswith('_outputs.'):
                    key = key.split('.')[1]

                id = record[self.databackend.id_field]
                assert not isinstance(vi.indexing_listener.model, str)
                h = record.outputs(
                    key,
                    vi.indexing_listener.model.identifier,
                    version=vi.indexing_listener.model.version,
                )
                if isinstance(h, Encodable):
                    h = h.x
                items.append(VectorItem.create(id=str(id), vector=h))

            searcher.add(items)
            progress.update(len(items))

    def set_compute(self, new: ComputeBackend):
        """
        Set a new compute engine at runtime. Use it only if you know what you do.
        The standard procedure is to set compute engine during initialization.
        """
        logging.warn(
            f"Change compute engine from '{self.compute.name}' to '{new.name}'"
        )

        self.compute.disconnect()
        logging.success(
            f"Succesfully disconnected from compute engine: '{self.compute.name}'"
        )

        logging.info(f"Connecting to compute engine: {new.name}")
        self.compute = new

    def get_compute(self):
        return self.compute

    def drop(self, force: bool = False):
        """
        Drop all data, artifacts and metadata
        """
        if not force and not click.confirm(
            f'{Colors.RED}[!!!WARNING USE WITH CAUTION AS YOU '
            f'WILL LOSE ALL DATA!!!]{Colors.RESET} '
            'Are you sure you want to drop the database? ',
            default=False,
        ):
            logging.warn('Aborting...')

        self.databackend.drop(force=True)
        self.metadata.drop(force=True)
        self.artifact_store.drop(force=True)

    def validate(
        self,
        identifier: str,
        type_id: str,
        validation_set: str,
        metrics: t.Sequence[str],
    ):
        """
        Evaluate quality of component, using `Component.validate`, if implemented.

        :param identifier: identifier of semantic index
        :param type_id: type_id of component
        :param validation_set: validation dataset on which to validate
        :param metrics: metric functions to compute
        """
        # TODO: never called
        component = self.load(type_id, identifier)
        metric_list = [self.load('metric', m) for m in metrics]
        assert isinstance(component, Model)
        return component.validate(
            self,
            validation_set,
            metric_list,  # type: ignore[arg-type]
        )

    def show(
        self,
        type_id: str,
        identifier: t.Optional[str] = None,
        version: t.Optional[int] = None,
    ):
        """
        Show available functionality which has been added using ``self.add``.
        If version is specified, then print full metadata

        :param type_id: type_id of component to show ['encoder', 'model', 'listener',
                       'learning_task', 'training_configuration', 'metric',
                       'vector_index', 'job']
        :param identifier: identifying string to component
        :param version: (optional) numerical version - specify for full metadata
        """
        if identifier is None and version is not None:
            raise ValueError(f'must specify {identifier} to go with {version}')

        if identifier is None:
            return self.metadata.show_components(type_id=type_id)

        if version is None:
            return self.metadata.show_component_versions(
                type_id=type_id, identifier=identifier
            )

        if version == -1:
            return self._get_object_info(type_id=type_id, identifier=identifier)

        return self._get_object_info(
            type_id=type_id, identifier=identifier, version=version
        )

    def _get_context(
        self, model, context_select: t.Optional[Select], context_key: t.Optional[str]
    ):
        assert model.takes_context, 'model does not take context'
        assert context_select is not None
        context = list(self.execute(context_select))
        context = [x.unpack() for x in context]

        if context_key is not None:
            context = [MongoStyleDict(x)[context_key] for x in context]
        return context

    async def apredict(
        self,
        model_name: str,
        input: t.Union[Document, t.Any],
        context_select: t.Optional[Select] = None,
        context_key: t.Optional[str] = None,
        **kwargs,
    ):
        """
        Apply model to input using asyncio.

        :param model_name: model identifier
        :param input: input to be passed to the model.
                      Must be possible to encode with registered encoders
        :param context_select: select query object to provide context
        :param context_key: key to use to extract context from context_select
        """
        model = self.models[model_name]
        context = None

        if context_select is not None:
            context = self._get_context(model, context_select, context_key)
        out = await model.apredict(
            input.unpack() if isinstance(input, Document) else input,
            one=True,
            context=context,
            **kwargs,
        )

        if model.encoder is not None:
            out = model.encoder(out)

        if context is not None:
            return Document(out), [Document(x) for x in context]
        return Document(out), []

    def predict(
        self,
        model_name: str,
        input: t.Union[Document, t.Any],
        context_select: t.Optional[t.Union[str, Select]] = None,
        context_key: t.Optional[str] = None,
        **kwargs,
    ) -> t.Tuple[Document, t.List[Document]]:
        """
        Apply model to input.

        :param model_name: model identifier
        :param input: input to be passed to the model.
                      Must be possible to encode with registered encoders
        :param context_select: select query object to provide context
        :param context_key: key to use to extract context from context_select
        """
        model = self.models[model_name]
        context = None

        if context_select is not None:
            if isinstance(context_select, Select):
                context = self._get_context(model, context_select, context_key)
            elif isinstance(context_select, str):
                context = context_select
            else:
                raise TypeError("context_select should be either Select or str")

        out = model.predict(
            input.unpack() if isinstance(input, Document) else input,
            one=True,
            context=context,
            **kwargs,
        )

        if isinstance(model.encoder, Encoder):
            out = model.encoder(out)

        if context is not None:
            return Document(out), [Document(x) for x in context]
        return Document(out), []

    def execute(self, query: ExecuteQuery, *args, **kwargs) -> ExecuteResult:
        """
        Execute a query on the db.

        :param query: select, insert, delete, update,
        """

        if isinstance(query, Delete):
            return self.delete(query, *args, **kwargs)
        if isinstance(query, Insert):
            return self.insert(query, *args, **kwargs)
        if isinstance(query, Select):
            return self.select(query, *args, **kwargs)
        if isinstance(query, Table):
            return self.select(query.to_query(), *args, **kwargs)
        if isinstance(query, Update):
            return self.update(query, *args, **kwargs)
        if isinstance(query, RawQuery):
            return query.execute(self)

        raise TypeError(
            f'Wrong type of {query}; '
            f'Expected object of type {t.Union[Select, Delete, Update, Insert, str]}; '
            f'Got {type(query)};'
        )

    def delete(self, delete: Delete, refresh: bool = True) -> DeleteResult:
        """
        Delete data.

        :param delete: delete query object
        """
        result = delete.execute(self)
        if refresh and not self.cdc.running:
            return result, self.refresh_after_delete(delete, ids=result)
        return result, None

    def insert(
        self, insert: Insert, refresh: bool = True, encoders: t.Sequence[Encoder] = ()
    ) -> InsertResult:
        """
        Insert data.

        :param insert: insert query object
        """
        for e in encoders:
            self.add(e)
        for r in insert.documents:
            r['_fold'] = 'train'  # type: ignore[assignment]
            if random.random() < s.CFG.fold_probability:
                r['_fold'] = 'valid'  # type: ignore[assignment]
        inserted_ids = insert.execute(self)

        if refresh and self.cdc.running:
            raise Exception('cdc cannot be activated and refresh=True')

        if s.CFG.cluster.cdc is not None:
            logging.info('CDC active, skipping refresh')
            return inserted_ids, None

        if refresh:
            return inserted_ids, self.refresh_after_update_or_insert(
                insert, ids=inserted_ids, verbose=False
            )
        return inserted_ids, None

    def select(self, select: Select) -> SelectResult:
        """
        Select data.

        :param select: select query object
        """
        if select.variables:
            select = select.set_variables(self)  # type: ignore[assignment]
        return select.execute(self)

    def refresh_after_delete(
        self,
        query: Delete,
        ids: t.Sequence[str],
        verbose: bool = False,
    ):
        """
        Trigger cleanup jobs after data deletion.

        :param query: Select or Update which reduces scope of computations
        :param ids: ids which reduce scopy of computations
        :param verbose: Toggle to ``True`` to get more output
        """
        task_workflow: TaskWorkflow = self._build_delete_task_workflow(
            query,
            ids=ids,
            verbose=verbose,
        )
        task_workflow.run_jobs()
        return task_workflow

    def refresh_after_update_or_insert(
        self,
        query: t.Union[Insert, Select, Update],
        ids: t.Sequence[str],
        verbose: bool = False,
    ):
        """
        Trigger computation jobs after data insertion.

        :param query: Select or Update which reduces scope of computations
        :param ids: ids which reduce scopy of computations
        :param verbose: Toggle to ``True`` to get more output
        """
        task_workflow: TaskWorkflow = self._build_task_workflow(
            query.select_table,  # TODO can be replaced by select_using_ids
            ids=ids,
            verbose=verbose,
        )
        task_workflow.run_jobs()
        return task_workflow

    def update(self, update: Update, refresh: bool = True) -> UpdateResult:
        """
        Update data.

        :param update: update query object
        """
        updated_ids = update.execute(self)

        if refresh and self.cdc.running:
            raise Exception('cdc cannot be activated and refresh=True')
        if refresh:
            return updated_ids, self.refresh_after_update_or_insert(
                query=update, ids=updated_ids, verbose=False
            )
        return updated_ids, None

    def add(
        self,
        object: t.Union[Component, t.Sequence[t.Any], t.Any],
        dependencies: t.Sequence[Job] = (),
    ):
        """
        Add functionality in the form of components. Components are stored in the
        configured artifact store, and linked to the primary db through
        the metadata.

        :param object: Object to be stored
        :param dependencies: list of jobs which should execute before component
                             init begins
        """
        if isinstance(object, (list, tuple)):
            return type(object)(
                self._add(
                    object=component,
                    dependencies=dependencies,
                )
                for component in object
            )
        elif isinstance(object, Component):
            return self._add(object=object, dependencies=dependencies), object
        else:
            return self._add(superduper(object)), object

    def remove(
        self,
        type_id: str,
        identifier: str,
        version: t.Optional[int] = None,
        force: bool = False,
    ):
        """
        Remove component (version: optional)

        :param type_id: type_id of component to remove ['encoder', 'model', 'listener',
                        'training_configuration', 'vector_index']
        :param identifier: identifier of component (see `container.base.Component`)
        :param version: [optional] numerical version to remove
        :param force: force skip confirmation (use with caution)
        """
        # TODO: versions = [version] if version is not None else ...
        if version is not None:
            return self._remove_component_version(
                type_id, identifier, version=version, force=force
            )
        versions = self.metadata.show_component_versions(type_id, identifier)
        versions_in_use = []
        for v in versions:
            if self.metadata.component_version_has_parents(type_id, identifier, v):
                versions_in_use.append(v)

        if versions_in_use:
            component_versions_in_use = []
            for v in versions_in_use:
                unique_id = Component.make_unique_id(type_id, identifier, v)
                component_versions_in_use.append(
                    f"{unique_id} -> "
                    f"{self.metadata.get_component_version_parents(unique_id)}",
                )
            if not force:
                raise exceptions.ComponentInUseError(
                    f'Component versions: {component_versions_in_use} are in use'
                )
            else:
                warnings.warn(
                    exceptions.ComponentInUseWarning(
                        f'Component versions: {component_versions_in_use}'
                        ', marking as hidden'
                    )
                )

        if force or click.confirm(
            f'You are about to delete {type_id}/{identifier}, are you sure?',
            default=False,
        ):
            for v in sorted(list(set(versions) - set(versions_in_use))):
                self._remove_component_version(type_id, identifier, v, force=True)

            for v in sorted(versions_in_use):
                self.metadata.hide_component_version(type_id, identifier, v)
        else:
            logging.warn('aborting.')

    def load(
        self,
        type_id: str,
        identifier: str,
        version: t.Optional[int] = None,
        allow_hidden: bool = False,
        info_only: bool = False,
    ) -> t.Union[Component, t.Dict[str, t.Any]]:
        """
        Load component using uniquely identifying information.

        :param type_id: type_id of component to remove
                        ['encoder', 'model', 'listener', ...]
        :param identifier: identifier of component (see `container.base.Component`)
        :param version: [optional] numerical version
        :param allow_hidden: toggle to ``True`` to allow loading of deprecated
                             components
        :param info_only: toggle to ``True`` to return metadata only
        """
        info = self.metadata.get_component(
            type_id=type_id,
            identifier=identifier,
            version=version,
            allow_hidden=allow_hidden,
        )

        if info is None:
            raise exceptions.MetadataException(
                f'No such object of type "{type_id}", '
                f'"{identifier}" has been registered.'
            )

        if info_only:
            return info

        def get_children(info):
            return {
                k: v
                for k, v in info['dict'].items()
                if isinstance(v, dict) and serializable.is_component_metadata(v)
            }

        def replace_children(r):
            if isinstance(r, dict):
                children = get_children(r)
                for k, v in children.items():
                    c = replace_children(
                        self.metadata.get_component(**v, allow_hidden=True)
                    )
                    try:
                        r['dict'][k] = c
                    except KeyError:
                        raise exceptions.MetadataException(
                            'Children {k} not found in `dict`'
                        )
            return r

        info = replace_children(info)
        info = self.artifact_store.load(info, lazy=True)

        m = Component.deserialize(info)
        m.on_load(self)

        if cm := self.type_id_to_cache_mapping.get(type_id):
            try:
                getattr(self, cm)[m.identifier] = m
            except KeyError:
                raise exceptions.ComponentException('%s not found in %s cache'.format())
        return m

    def _build_delete_task_workflow(
        self,
        query: Delete,
        ids: t.Sequence[str],
        verbose: bool = False,
    ):
        G = TaskWorkflow(self)
        vector_indices = self.show('vector_index')

        if not vector_indices:
            return G

        deleted_table_or_collection = query.table_or_collection.identifier

        for vi in vector_indices:
            vi = self.vector_indices[vi]
            listener_table_or_collection = (
                vi.indexing_listener.select.table_or_collection.identifier
            )

            if deleted_table_or_collection != listener_table_or_collection:
                continue

            if (
                s.CFG.vector_search == 'in_memory'
                and vi.identifier not in self.fast_vector_searchers
            ):
                continue

            G.add_node(
                f'{vi.identifier}.{delete_vectors.__name__}()',
                job=FunctionJob(
                    callable=delete_vectors,
                    args=[],
                    kwargs=dict(
                        vector_index=vi.identifier,
                        ids=ids,
                    ),
                ),
            )

        return G

    def _build_task_workflow(
        self,
        query,
        ids=None,
        dependencies=(),
        verbose: bool = True,
    ) -> TaskWorkflow:
        logging.debug(f"Building task workflow graph. Query:{query}")

        job_ids: t.Dict[str, t.Any] = defaultdict(lambda: [])
        job_ids.update(dependencies)
        # TODO use these job_ids as dependencies for every job

        G = TaskWorkflow(self)

        # TODO extract this logic from this class
        G.add_node(
            f'{download_content.__name__}()',
            job=FunctionJob(
                callable=download_content,
                args=[],
                kwargs=dict(
                    ids=ids,
                    query=query.serialize(),
                ),
            ),
        )

        listeners = self.show('listener')
        if not listeners:
            return G
        listener_selects = {}

        for identifier in listeners:
            info = self.metadata.get_component('listener', identifier)
            listener_query = info['dict']['select']

            listener_select = serializable.Serializable.deserialize(listener_query)
            listener_selects.update({identifier: listener_select})
            if listener_select is None:
                continue
            if (
                listener_select.table_or_collection.identifier
                != query.table_or_collection.identifier
            ):
                continue

            model, _, key = identifier.rpartition('/')
            G.add_node(
                f'{model}.predict({key})',
                job=ComponentJob(
                    component_identifier=model,
                    args=[key],
                    kwargs={
                        'ids': ids,
                        'select': listener_query,
                        **info['dict']['predict_kwargs'],
                    },
                    method_name='predict',
                    type_id='model',
                ),
            )

        for identifier in listeners:
            listener_select = listener_selects[identifier]
            if listener_select is None:
                continue
            if (
                listener_select.table_or_collection.identifier
                != query.table_or_collection.identifier
            ):
                continue
            model, _, key = identifier.rpartition('/')
            G.add_edge(
                f'{download_content.__name__}()',
                f'{model}.predict({key})',
            )
            deps = self._get_dependencies_for_listener(identifier)
            for dep in deps:
                dep_model, _, dep_key = dep.rpartition('/')
                G.add_edge(
                    f'{dep_model}.predict({dep_key})',
                    f'{model}.predict({key})',
                )

        if s.CFG.self_hosted_vector_search:
            return G

        for identifier in self.show('vector_index'):
            # if a vector-searcher is not loaded, then skip
            # since s.CFG.vector_search == 'in_memory' implies the
            # program is standalone

            if (
                s.CFG.vector_search == 'in_memory'
                and identifier not in self.fast_vector_searchers
            ):
                continue

            vi = self.vector_indices[identifier]
            if (
                vi.indexing_listener.select.table_or_collection.identifier
                != query.table_or_collection.identifier
            ):
                continue

            G.add_node(
                f'{identifier}.{copy_vectors.__name__}',
                FunctionJob(
                    callable=copy_vectors,
                    args=[],
                    kwargs={
                        'vector_index': identifier,
                        'ids': ids,
                        'query': vi.indexing_listener.select.serialize(),
                    },
                ),
            )
            model = vi.indexing_listener.model.identifier
            key = vi.indexing_listener.key
            G.add_edge(
                f'{model}.predict({key})', f'{identifier}.{copy_vectors.__name__}'
            )

        return G

    def _compute_model_outputs(
        self,
        model_info,
        _ids,
        select: Select,
        key='_base',
        model=None,
        predict_kwargs=None,
    ):
        s.logging.info('finding documents under filter')
        model_identifier = model_info['identifier']
        documents = list(self.execute(select.select_using_ids(_ids)))
        documents = [x.unpack() for x in documents]
        if key != '_base':
            passed_docs = [r[key] for r in documents]
        else:
            passed_docs = documents
        if model is None:
            model = self.models[model_identifier]
        return model.predict(passed_docs, **(predict_kwargs or {}))

    def _save_artifacts(self, artifact_dictionary: t.Dict):
        raise NotImplementedError  # TODO use in the server code...

    def _add(
        self,
        object: Component,
        dependencies: t.Sequence[Job] = (),
        serialized: t.Optional[t.Dict] = None,
        parent: t.Optional[str] = None,
    ):
        jobs = []
        object.pre_create(self)
        assert hasattr(object, 'identifier')
        assert hasattr(object, 'version')

        existing_versions = self.show(object.type_id, object.identifier)
        if isinstance(object.version, int) and object.version in existing_versions:
            s.logging.debug(f'{object.unique_id} already exists - doing nothing')
            return []

        if existing_versions:
            object.version = max(existing_versions) + 1
        else:
            object.version = 0

        if serialized is None:
            serialized, artifacts = object.serialized
            artifact_info = self.artifact_store.save(artifacts)
            serialized = self.artifact_store.replace(serialized, artifact_info)

        else:
            try:
                serialized['version'] = object.version
                serialized['dict']['version'] = object.version
            except KeyError:
                raise exceptions.MetadataException(
                    '`dict` or `version` not found in serialized dict.'
                )

        jobs.extend(self._create_children(object, serialized))
        self.metadata.create_component(serialized)
        if parent is not None:
            self.metadata.create_parent_child(parent, object.unique_id)
        object.post_create(self)
        these_jobs = object.schedule_jobs(self, dependencies=dependencies)
        jobs.extend(these_jobs)
        return jobs

    def _create_children(self, component: Component, serialized: t.Dict):
        jobs = []
        for k, child_type_id in component.child_components:
            assert isinstance(k, str)
            child = getattr(component, k)
            if isinstance(child, str):
                serialized_dict = {
                    'type_id': child_type_id,
                    'identifier': child,
                    'version': self.metadata.get_latest_version(child_type_id, child),
                }
                self.metadata.create_parent_child(
                    component.unique_id, Component.make_unique_id(**serialized_dict)
                )
            else:
                sub_jobs = self._add(
                    child,
                    serialized=serialized['dict'][k],
                    parent=component.unique_id,
                )
                jobs.extend(sub_jobs)
                serialized_dict = {
                    'type_id': child.type_id,
                    'identifier': child.identifier,
                    'version': child.version,
                }
            serialized['dict'][k] = serialized_dict

        return jobs

    def _create_plan(self):
        G = networkx.DiGraph()
        for identifier in self.metadata.show_components('listener', active=True):
            G.add_node('listener', job=identifier)
        for identifier in self.metadata.show_components('listener'):
            deps = self._get_dependencies_for_listener(identifier)
            for dep in deps:
                G.add_edge(('listener', dep), ('listener', identifier))
        if not networkx.is_directed_acyclic_graph(G):
            raise ValueError('G is not a directed, acyclic graph')
        return G

    def _delete(self, delete: Delete):
        delete.execute(self)

    def _remove_component_version(
        self,
        type_id: str,
        identifier: str,
        version: int,
        force: bool = False,
    ):
        unique_id = Component.make_unique_id(type_id, identifier, version)
        if self.metadata.component_version_has_parents(type_id, identifier, version):
            parents = self.metadata.get_component_version_parents(unique_id)
            raise Exception(f'{unique_id} is involved in other components: {parents}')

        if force or click.confirm(
            f'You are about to delete {unique_id}, are you sure?',
            default=False,
        ):
            component = self.load(type_id, identifier, version=version)
            info = self.metadata.get_component(type_id, identifier, version=version)
            if hasattr(component, 'cleanup'):
                component.cleanup(self)
            if type_id in self.type_id_to_cache_mapping:
                try:
                    del getattr(self, self.type_id_to_cache_mapping[type_id])[
                        identifier
                    ]
                except KeyError:
                    pass

            if hasattr(component, 'artifact_attributes'):
                for a in component.artifact_attributes:
                    self.artifact_store.delete(info['dict'][a]['file_id'])
            self.metadata.delete_component_version(type_id, identifier, version=version)

    def _download_content(  # TODO: duplicated function
        self,
        query: t.Optional[t.Union[Select, Insert]] = None,
        ids=None,
        documents=None,
        timeout=None,
        raises: bool = True,
        n_download_workers=None,
        headers=None,
        **kwargs,
    ):
        update_db = False

        if documents is not None:
            pass
        elif isinstance(query, Select):
            update_db = True
            if ids is None:
                documents = list(self.select(query))
            else:
                select = query.select_using_ids(ids)
                cursor = self.select(select).raw_cursor
                documents = [Document(x) for x in cursor]
        elif isinstance(query, Insert):
            documents = query.documents
        else:
            raise NotImplementedError

        documents = [x.content for x in documents]
        uris, keys, place_ids = gather_uris(documents)
        s.logging.info(f'found {len(uris)} uris')
        if not uris:
            return

        if n_download_workers is None:
            try:
                n_download_workers = self.metadata.get_metadata(
                    key='n_download_workers'
                )
            except exceptions.MetadatastoreException:
                n_download_workers = 0

        if headers is None:
            try:
                headers = self.metadata.get_metadata(key='headers')
            except exceptions.MetadatastoreException:
                headers = 0

        if timeout is None:
            try:
                timeout = self.metadata.get_metadata(key='download_timeout')
            except exceptions.MetadatastoreException:
                timeout = None

        def download_update(key, id, bytes):
            return query.download_update(db=self, key=key, id=id, bytes=bytes)

        downloader = Downloader(
            uris=uris,
            ids=place_ids,
            keys=keys,
            update_one=download_update,
            n_workers=n_download_workers,
            timeout=timeout,
            headers=headers,
            raises=raises,
        )
        downloader.go()
        if update_db:
            return
        for id_, key in zip(place_ids, keys):
            documents[id_] = self.databackend.set_content_bytes(
                documents[id_], key, downloader.results[id_]  # type: ignore[index]
            )
        return documents

    def _get_content_for_filter(self, filter) -> Document:
        if isinstance(filter, dict):
            filter = Document(content=filter)
        if '_id' not in filter.content:
            filter['_id'] = 0
        uris = gather_uris([filter.content])[0]
        if uris:
            output = self._download_content(
                query=None, documents=[filter.content], timeout=None, raises=True
            )[0]
            filter = Document(Document.decode(output, encoders=self.encoders))
        if not filter['_id']:
            del filter.content['_id']
        return filter

    def _get_dependencies_for_listener(self, identifier):
        info = self.metadata.get_component('listener', identifier)
        if info is None:
            return []
        out = []
        if info['dict']['key'].startswith('_outputs.'):
            _, key, model = info['dict']['key'].split('.')
            out.append(f'{model}/{key}')
        return out

    def _get_file_content(self, r):
        for k in r:
            if isinstance(r[k], dict):
                r[k] = self._get_file_content(r[k])
        return r

    def _get_object_info(self, identifier, type_id, version=None):
        return self.metadata.get_component(type_id, identifier, version=version)

    def _apply_listener(
        self,
        identifier,
        ids: t.Optional[t.Sequence[str]] = None,
        verbose: bool = False,
        max_chunk_size=5000,
        model=None,
        recompute: bool = False,
        listener_info=None,
        **kwargs,
    ) -> t.List:
        # NOTE: this method is never called anywhere except for itself!
        if listener_info is None:
            listener_info = self.metadata.get_component('listener', identifier)

        select = serializable.Serializable.deserialize(listener_info['select'])
        if ids is None:
            ids = select.get_ids(self)
        else:
            ids = select.select_using_ids(ids=ids).get_ids(self)
            assert ids is not None

        ids = [str(id) for id in ids]

        if max_chunk_size is not None:
            for it, i in enumerate(range(0, len(ids), max_chunk_size)):
                s.logging.info(
                    'computing chunk '
                    f'({it + 1}/{math.ceil(len(ids) / max_chunk_size)})'
                )
                self._apply_listener(
                    identifier,
                    ids=ids[i : i + max_chunk_size],
                    verbose=verbose,
                    max_chunk_size=None,
                    model=model,
                    recompute=recompute,
                    listener_info=listener_info,
                    **kwargs,
                )
            return []

        model_info = self.metadata.get_component('model', listener_info['model'])
        outputs = self._compute_model_outputs(
            model_info,
            ids,
            select,
            key=listener_info['key'],
            model=model,
            predict_kwargs=listener_info.get('predict_kwargs', {}),
        )
        type = model_info.get('type')
        if type is not None:
            type = self.encoders[type]
            outputs = [type(x).encode() for x in outputs]

        select.model_update(
            db=self,
            model=listener_info['model'],
            key=listener_info['key'],
            version=listener_info['version'],
            outputs=outputs,
            ids=ids,
        )
        return outputs

    def replace(
        self,
        object: t.Any,
        upsert: bool = False,
    ):
        """
        (Use-with caution!!) Replace a model in artifact store with updated object.
        :param object: object to replace
        :param upsert: toggle to ``True`` to enable even if object doesn't exist yet
        """
        try:
            info = self.metadata.get_component(
                object.type_id, object.identifier, version=object.version
            )
        except FileNotFoundError:
            if upsert:
                return self.add(
                    object,
                )
            raise FileNotFoundError

        # If object has no version, update the last version
        object.version = info['version']
        new_info = self.artifact_store.update(object, metadata_info=info)
        self.metadata.replace_object(
            new_info,
            identifier=object.identifier,
            type_id='model',
            version=object.version,
        )

    def select_nearest(
        self,
        like: Document,
        vector_index: str,
        ids: t.Optional[t.Sequence[str]] = None,
        outputs: t.Optional[Document] = None,
        n: int = 100,
    ) -> t.Tuple[t.List[str], t.List[float]]:
        like = self._get_content_for_filter(like)
        vi = self.vector_indices[vector_index]
        if outputs is None:
            outs = {}
        else:
            outs = outputs.encode()
            if not isinstance(outs, dict):
                raise TypeError(f'Expected dict, got {type(outputs)}')
        return vi.get_nearest(like, db=self, ids=ids, n=n, outputs=outs)

    def close(self):
        """
        Gracefully shutdown the Datalayer
        """
        logging.info("Disconnect from Data Store")
        self.databackend.disconnect()

        logging.info("Disconnect from Metadata Store")
        self.metadata.disconnect()

        logging.info("Disconnect from Artifact Store")
        self.artifact_store.disconnect()

        logging.info("Disconnect from Compute Engine")
        self.compute.disconnect()

        # TODO: gracefully close all opened connections
        return


@dc.dataclass
class LoadDict(dict):
    database: Datalayer
    field: t.Optional[str] = None
    callable: t.Optional[t.Callable] = None

    def __missing__(self, key: str):
        if self.field is not None:
            value = self[key] = self.database.load(self.field, key)
        else:
            msg = f'callable is ``None`` for {key}'
            assert self.callable is not None, msg
            value = self[key] = self.callable(key)
        return value
