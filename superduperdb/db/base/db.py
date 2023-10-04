import dataclasses as dc
import math
import typing as t
import warnings
from collections import defaultdict

import click
import networkx
from dask.distributed import Future

import superduperdb as s
from superduperdb.container.artifact_tree import (
    get_artifacts,
    infer_artifacts,
    load_artifacts_from_store,
    replace_artifacts,
)
from superduperdb.container.component import Component
from superduperdb.container.document import Document
from superduperdb.container.job import ComponentJob, FunctionJob, Job
from superduperdb.container.model import Model
from superduperdb.container.serializable import Serializable
from superduperdb.container.task_workflow import TaskWorkflow
from superduperdb.db.base.download import Downloader, gather_uris
from superduperdb.misc.colors import Colors
from superduperdb.misc.special_dicts import MongoStyleDict
from superduperdb.vector_search.base import VectorDatabase

from .artifact import ArtifactStore
from .cursor import SuperDuperCursor
from .data_backend import BaseDataBackend
from .download_content import download_content
from .exceptions import ComponentInUseError, ComponentInUseWarning
from .metadata import MetaDataStore
from .query import Delete, Insert, Like, Select, SelectOne, Update

DBResult = t.Any
TaskGraph = t.Any

DeleteResult = DBResult
InsertResult = t.Tuple[DBResult, t.Optional[TaskGraph]]
SelectResult = SuperDuperCursor
UpdateResult = t.Any

ExecuteQuery = t.Union[Select, SelectOne, Delete, Update, Insert]
ExecuteResult = t.Union[SelectResult, DeleteResult, UpdateResult, InsertResult]

ENDPOINTS = 'delete', 'execute', 'insert', 'like', 'select', 'select_one', 'update'


class DB:
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
        vector_database: t.Optional[VectorDatabase] = None,
        distributed_client=None,
    ):
        """
        :param databackend: databackend object containing connection to Datastore
        :param metadata: metadata object containing connection to Metadatastore
        :param artifact_store: artifact_store object containing connection to
                               Artifactstore
        :param vector_database: vector_database object containing connection to
                                VectorDatabase
        :param distributed_client:
        """
        self.metrics = LoadDict(self, 'metric')
        self.models = LoadDict(self, 'model')
        self.encoders = LoadDict(self, 'encoder')
        self.vector_indices = LoadDict(self, 'vector_index')

        self.distributed = s.CFG.cluster.distributed
        self.metadata = metadata
        self.artifact_store = artifact_store
        self.databackend = databackend
        self.vector_database = vector_database
        self._distributed_client = distributed_client

    @property
    def db(self):
        return self.databackend.db

    @property
    def distributed_client(self):
        return self._distributed_client

    def create_output_table(self, *args):
        """
        Create output table for a model, This is valid for sql databases,
        and databases with seperate output table configuration.
        """
        pass

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
            print('Aborting...')

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

        :param model: model identifier
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
        context_select: t.Optional[Select] = None,
        context_key: t.Optional[str] = None,
        **kwargs,
    ) -> t.Tuple[Document, t.List[Document]]:
        """
        Apply model to input.

        :param model: model identifier
        :param input: input to be passed to the model.
                      Must be possible to encode with registered encoders
        :param context_select: select query object to provide context
        :param context_key: key to use to extract context from context_select
        """
        model = self.models[model_name]
        context = None

        if context_select is not None:
            context = self._get_context(model, context_select, context_key)

        out = model.predict(
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

    def execute(self, query: ExecuteQuery) -> ExecuteResult:
        """
        Execute a query on the db.

        :param query: select, insert, delete, update,
        """
        if isinstance(query, Delete):
            return self.delete(query)
        if isinstance(query, Insert):
            return self.insert(query)
        if isinstance(query, Select):
            return self.select(query)
        if isinstance(query, Like):
            return self.like(query)
        if isinstance(query, SelectOne):
            return self.select_one(query)
        if isinstance(query, Update):
            return self.update(query)
        raise TypeError(
            f'Wrong type of {query}; '
            f'Expected object of type {t.Union[Select, Delete, Update, Insert]}; '
            f'Got {type(query)};'
        )

    def delete(self, delete: Delete) -> DeleteResult:
        """
        Delete data.

        :param delete: delete query object
        """
        return delete(self)

    def insert(self, insert: Insert) -> InsertResult:
        """
        Insert data.

        :param insert: insert query object
        """
        return insert(self)

    def run(
        self,
        job,
        depends_on: t.Optional[t.Sequence[Future]] = None,
        distributed: t.Optional[bool] = None,
    ):
        """
        Run job. See ``container.job.Job``, ``container.job.FunctionJob``,
        ``container.job.ComponentJob``.

        :param job:
        :param depends_on: t.Sequence of dependencies
        """
        if distributed is None:
            distributed = s.CFG.cluster.distributed
        return job(db=self, dependencies=depends_on, distributed=distributed)

    def select(self, select: Select) -> SelectResult:
        """
        Select data.

        :param select: select query object
        """
        return select(self)

    def like(self, like: Like) -> SelectResult:
        """
        Perform vector search over data.

        :param like: like query object
        """
        return like(self)

    def select_one(self, select_one: SelectOne) -> SelectResult:
        """
        Select data and return a single result.

        :param select_one: select-a-single document query object
        """
        return select_one(self)

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
            query.select_table,
            ids=ids,
            verbose=verbose,
        )
        task_workflow.run_jobs(distributed=self.distributed)
        return task_workflow

    def update(self, update: Update) -> UpdateResult:
        """
        Update data.

        :param update: update query object
        """
        return update(self)

    def add(
        self,
        object: Component,
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
        # TODO why this helper function?
        return self._add(
            object=object,
            dependencies=dependencies,
        )

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
        if version is not None:
            return self._remove_component_version(type_id, identifier, version=version)
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
            f'You are about to delete {type_id}/{identifier}, are you sure?',
            default=False,
        ):
            for v in sorted(list(set(versions) - set(versions_in_use))):
                self._remove_component_version(type_id, identifier, v, force=True)

            for v in sorted(versions_in_use):
                self.metadata.hide_component_version(type_id, identifier, v)
        else:
            print('aborting.')

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

        :param type_id: type_id of component to remove ['encoder', 'model', 'listener',
                        'training_configuration', 'learning_task', 'vector_index']
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
            raise Exception(
                f'No such object of type "{type_id}", '
                f'"{identifier}" has been registered.'
            )

        if info_only:
            return info

        def get_children(info):
            return {
                k: v
                for k, v in info['dict'].items()
                if isinstance(v, dict)
                and set(v.keys()) == {'type_id', 'identifier', 'version'}
            }

        def replace_children(r):
            if isinstance(r, dict):
                children = get_children(r)
                for k, v in children.items():
                    r['dict'][k] = replace_children(
                        self.metadata.get_component(**v, allow_hidden=True)
                    )
            return r

        info = replace_children(info)
        info = load_artifacts_from_store(
            info, cache={}, artifact_store=self.artifact_store
        )

        m = Component.deserialize(info)
        m.on_load(self)

        if cm := self.type_id_to_cache_mapping.get(type_id):
            getattr(self, cm)[m.identifier] = m
        return m

    def _build_task_workflow(
        self,
        query,
        ids=None,
        dependencies=(),
        verbose: bool = True,
    ) -> TaskWorkflow:
        job_ids: t.Dict[str, t.Any] = defaultdict(lambda: [])
        job_ids.update(dependencies)
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
        listener = self.show('listener')
        if not listener:
            return G

        for identifier in listener:
            info = self.metadata.get_component('listener', identifier)
            query = info['dict']['select']
            model, _, key = identifier.rpartition('/')
            G.add_node(
                f'{model}.predict({key})',
                job=ComponentJob(
                    component_identifier=model,
                    args=[key],
                    kwargs={
                        'ids': ids,
                        'select': query,
                        **info['dict']['predict_kwargs'],
                    },
                    method_name='predict',
                    type_id='model',
                ),
            )

        for identifier in listener:
            model, _, key = identifier.rpartition('/')
            G.add_edge(
                f'{download_content.__name__}()',
                f'{model}.predict({key})',
            )
            deps = self._get_dependencies_for_listener(
                identifier
            )  # TODO remove features as explicit argument to listener
            for dep in deps:
                dep_model, _, dep_key = dep.rpartition('/')
                G.add_edge(
                    f'{dep_model}.predict({dep_key})',
                    f'{model}.predict({key})',
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
        s.log.info('finding documents under filter')
        features = features or {}
        model_identifier = model_info['identifier']
        if features is None:
            features = {}
        documents = list(self.execute(select.select_using_ids(_ids)))
        s.log.info('done.')
        documents = [x.unpack() for x in documents]
        if key != '_base' or '_base' in features:
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
        object.on_create(self)
        assert hasattr(object, 'identifier')
        assert hasattr(object, 'version')

        existing_versions = self.show(object.type_id, object.identifier)
        if isinstance(object.version, int) and object.version in existing_versions:
            s.log.warn(f'{object.unique_id} already exists - doing nothing')
            return

        if existing_versions:
            object.version = max(existing_versions) + 1
        else:
            object.version = 0

        if serialized is None:
            serialized = object.serialize()
            artifacts = list(set(get_artifacts(serialized)))
            artifact_info = {}
            for a in artifacts:
                artifact_info[hash(a)] = a.save(self.artifact_store)
            serialized = t.cast(t.Dict, replace_artifacts(serialized, artifact_info))

        else:
            serialized['version'] = object.version
            serialized['dict']['version'] = object.version

        self._create_children(object, serialized)

        self.metadata.create_component(serialized)
        if parent is not None:
            self.metadata.create_parent_child(parent, object.unique_id)
        object.on_load(
            self
        )  # TODO do I really need to call this here? Could be handled by `.on_create`?
        jobs = object.schedule_jobs(self, dependencies=dependencies)
        return jobs

    def _create_children(self, component: Component, serialized: t.Dict):
        for k, child_type_id in component.child_components:
            assert isinstance(k, str)
            child = getattr(component, k)
            if isinstance(child, str):
                serialized_dict = {
                    'type_id': child_type_id,
                    'identifier': child,
                    'version': self.metadata.get_latest_version(child_type_id, child),
                }
            else:
                self._add(
                    child,
                    serialized=serialized['dict'][k],
                    parent=component.unique_id,
                )
                serialized_dict = {
                    'type_id': child.type_id,
                    'identifier': child.identifier,
                    'version': child.version,
                }
            serialized['dict'][k] = serialized_dict

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
        return self.db.delete(delete)

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

            if hasattr(component, 'artifacts'):
                for a in component.artifacts:
                    self.artifact_store.delete_artifact(info['dict'][a]['file_id'])
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
        s.log.info(f'found {len(uris)} uris')
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
            documents[id_] = self.db.set_content_bytes(
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
        return filter

    def _get_dependencies_for_listener(self, identifier):
        info = self.metadata.get_component('listener', identifier)
        if info is None:
            return []
        listener_features = info.get('features', {})
        out = []
        for k in listener_features:
            out.append(f'{self.features[k]}/{k}')
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

        select = Serializable.deserialize(listener_info['select'])
        if ids is None:
            ids = select.get_ids(self)
        else:
            ids = select.select_using_ids(ids=ids).get_ids(self)
            assert ids is not None

        ids = [str(id) for id in ids]

        if max_chunk_size is not None:
            for it, i in enumerate(range(0, len(ids), max_chunk_size)):
                s.log.info(
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
                    distributed=False,
                    **kwargs,
                )
            return []

        model_info = self.metadata.get_component('model', listener_info['model'])
        outputs = self._compute_model_outputs(
            model_info,
            ids,
            select,
            key=listener_info['key'],
            features=listener_info.get('features', {}),
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
        except FileNotFoundError as e:
            if upsert:
                return self.add(
                    object,
                )
            raise e

        new_info = object.serialize()
        all_artifacts = tuple(set(get_artifacts(new_info)))
        artifact_details = dict()
        for a in all_artifacts:
            artifact_details[hash(a)] = a.save(self.artifact_store)

        old_artifacts = tuple(set(infer_artifacts(info)))
        for oa in old_artifacts:
            self.artifact_store.delete_artifact(oa)

        new_info = replace_artifacts(new_info, artifact_details)

        self.metadata.replace_object(
            new_info,
            identifier=object.identifier,
            type_id='model',
            version=object.version,
        )

    def _select_nearest(
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


@dc.dataclass
class LoadDict(dict):
    database: DB
    field: str

    def __missing__(self, key: str):
        value = self[key] = self.database.load(self.field, key)
        return value
