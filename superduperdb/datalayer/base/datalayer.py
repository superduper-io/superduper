from dask.distributed import Future
import dataclasses as dc
import math
import typing as t
import warnings
from collections import defaultdict
from typing import Union, Optional, List, Tuple

import click
import networkx

from superduperdb import CFG
from superduperdb.core.task_workflow import TaskWorkflow
from superduperdb.core.component import Component
from superduperdb.core.document import Document
from superduperdb.core.exceptions import ComponentInUseError, ComponentInUseWarning
from superduperdb.datalayer.base.artifacts import ArtifactStore
from superduperdb.datalayer.base.data_backend import BaseDataBackend
from superduperdb.datalayer.base.metadata import MetaDataStore
from superduperdb.datalayer.base.query import (
    Insert,
    Select,
    Delete,
    Update,
    SelectOne,
    Like,
)
from .download_content import download_content  # type: ignore[attr-defined]
from superduperdb.misc.downloads import Downloader
from superduperdb.misc.downloads import gather_uris
from superduperdb.misc.logger import logging
from superduperdb.vector_search.base import VectorDatabase

# TODO:
# This global variable is a temporary solution to make VectorDatabase available
# to the rest of the code.
# It should be moved to the Server's initialization code where it can be available to
# all threads.
from superduperdb.core.artifact import Artifact
from superduperdb.core.artifact_tree import (
    get_artifacts,
    infer_artifacts,
    put_artifacts_back,
    replace_artifacts,
)
from ...core.job import FunctionJob, ComponentJob, Job
from ...core.serializable import Serializable


DBResult = t.Any
TaskGraph = t.Any

DeleteResult = DBResult
InsertResult = t.Tuple[DBResult, t.Optional[TaskGraph]]
SelectResult = t.List[Document]
UpdateResult = t.Any

ExecuteQuery = t.Union[Select, Delete, Update, Insert]
ExecuteResult = t.Union[SelectResult, DeleteResult, UpdateResult, InsertResult]

ENDPOINTS = 'delete', 'execute', 'insert', 'like', 'select', 'select_one', 'update'


class Datalayer:
    """
    Base database connector for SuperDuperDB
    """

    _database_type: str
    name: str
    select_cls: t.Type[Select]
    models: t.Dict

    variety_to_cache_mapping = {
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
        :param databackend:
        :param metadata:
        :param artifact_store:
        :param vector_database:
        :param distributed_client:
        """
        self.metrics = LoadDict(self, 'metric')
        self.models = LoadDict(self, 'model')
        self.encoders = LoadDict(self, 'encoder')
        self.vector_indices = LoadDict(self, 'vector_index')

        self.distributed = CFG.distributed
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

    def validate(
        self,
        identifier: str,
        variety: str,
        validation_set: str,
        metrics: List[str],
    ):
        """
        Evaluate quality of component, using `Component.validate`, if implemented.

        :param identifier: identifier of semantic index
        :param variety: variety of component
        :param validation_set: validation dataset on which to validate
        :param metrics: metric functions to compute
        """
        component = self.load(variety, identifier)
        metrics = [self.load('metric', m) for m in metrics]  # type: ignore[misc]
        return component.validate(self, validation_set, metrics)  # type: ignore[attr-defined]

    def show(
        self,
        variety: str,
        identifier: Optional[str] = None,
        version: Optional[int] = None,
    ):
        """
        Show available functionality which has been added using ``self.add``.
        If version is specified, then print full metadata

        :param variety: variety of component to show ["encoder", "model", "watcher",
                       "learning_task", "training_configuration", "metric",
                       "vector_index", "job"]
        :param identifier: identifying string to component
        :param version: (optional) numerical version - specify for full metadata
        """
        if identifier is None and version is not None:
            raise ValueError(f'must specify {identifier} to go with {version}')

        if identifier is None:
            return self.metadata.show_components(variety=variety)

        if version is None:
            return self.metadata.show_component_versions(
                variety=variety, identifier=identifier
            )

        if version == -1:
            return self._get_object_info(variety=variety, identifier=identifier)

        return self._get_object_info(
            variety=variety, identifier=identifier, version=version
        )

    def predict(
        self,
        model_identifier: str,
        input: Document,
        one: bool = False,
    ) -> Union[List[Document], Document]:
        """
        Apply model to input.

        :param model_identifier: model or ``str`` referring to an uploaded model
        :param input: input to be passed to the model.
                      Must be possible to encode with registered encoders
        :param one: if True passed a single document else passed multiple documents
        """
        model = self.models[model_identifier]
        opts = self.metadata.get_component('model', model_identifier)
        out = model.predict(input.unpack(), **opts.get('predict_kwargs', {}), one=one)
        if one:
            if model.encoder is not None:
                out = model.encoder(out)  # type: ignore
            return Document(out)
        else:
            if model.encoder is not None:
                out = [model.encoder(x) for x in out]  # type: ignore
            return [Document(x) for x in out]

    def execute(self, query: ExecuteQuery) -> ExecuteResult:
        """
        Execute a query on the datalayer.

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
            f'Expected object of type {Union[Select, Delete, Update, Insert]}; '
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
        depends_on: t.Optional[t.List[Future]] = None,
        distributed: t.Optional[bool] = None,
    ):
        """
        Run job. See ``core.job.Job``, ``core.job.FunctionJob``, ``core.job.ComponentJob``.

        :param job:
        :param depends_on: List of dependencies
        """
        if distributed is None:
            distributed = CFG.distributed
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
        self, query: Union[Select, Update], ids: List[str], verbose=False
    ):
        """
        Trigger computation jobs after data insertion.

        :param query: Select or Update which reduces scope of computations
        :param ids: ids which reduce scopy of computations
        :param verbose: Toggle to ``True`` to get more output
        """
        task_graph: TaskWorkflow = self._build_task_workflow(
            query.select_table,
            ids=ids,
            verbose=verbose,
        )
        task_graph(db=self, distributed=self.distributed)
        return task_graph

    def update(self, update: Update) -> UpdateResult:
        """
        Update data.

        :param update: update query object
        """
        return update(self)

    def add(
        self,
        object: Component,
        dependencies: t.Sequence[Union[Job, str]] = (),
    ):
        """
        Add functionality in the form of components. Components are stored in the
        configured artifact store, and linked to the primary datalayer through
        the metadata.

        :param object: Object to be stored
        :param dependencies: list of jobs which should execute before component init begins
        """
        return self._add(
            object=object,
            dependencies=dependencies,
        )

    def remove(
        self,
        variety: str,
        identifier: str,
        version: Optional[int] = None,
        force=False,
    ):
        """
        Remove component (version: optional)

        :param variety: variety of component to remove ["encoder", "model", "watcher",
                        "training_configuration", "learning_task", "vector_index"]
        :param identifier: identifier of component (see `core.base.Component`)
        :param version: [optional] numerical version to remove
        :param force: force skip confirmation (use with caution)
        """
        if version is not None:
            return self._remove_component_version(variety, identifier, version=version)
        versions = self.metadata.show_component_versions(variety, identifier)
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
                self._remove_component_version(variety, identifier, v, force=True)

            for v in sorted(versions_in_use):
                self.metadata.hide_component_version(variety, identifier, v)
        else:
            print('aborting.')

    def load(
        self,
        variety: str,
        identifier: str,
        version: Optional[int] = None,
        allow_hidden: bool = False,
        info_only: bool = False,
    ) -> Component:
        """
        Load component using uniquely identifying information.

        :param variety: variety of component to remove ["encoder", "model", "watcher",
                        "training_configuration", "learning_task", "vector_index"]
        :param identifier: identifier of component (see `core.base.Component`)
        :param version: [optional] numerical version
        :param repopulate: toggle to ``False`` to only load references to other
                           components
        :param allow_hidden: toggle to ``True`` to allow loading of deprecated
                             components
        """
        info = self.metadata.get_component(
            variety=variety,
            identifier=identifier,
            version=version,
            allow_hidden=allow_hidden,
        )

        if info is None:
            raise Exception(
                f'No such object of type "{variety}", '
                f'"{identifier}" has been registered.'
            )

        if info_only:
            return info

        def get_children(info):
            return {
                k: v
                for k, v in info['dict'].items()
                if isinstance(v, dict)
                and set(v.keys()) == {'variety', 'identifier', 'version'}
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
        info = put_artifacts_back(info, lookup={}, artifact_store=self.artifact_store)

        m = Component.deserialize(info)
        m._on_load(self)

        if cm := self.variety_to_cache_mapping.get(variety):
            getattr(self, cm)[m.identifier] = m
        return m

    def _build_task_workflow(
        self,
        query,
        ids=None,
        dependencies=(),
        verbose=True,
    ) -> TaskWorkflow:
        job_ids: t.Dict[str, t.Any] = defaultdict(lambda: [])
        job_ids.update(dependencies)
        G = TaskWorkflow(self)

        # TODO extract this logic from this class
        G.add_node(
            f'{download_content.__name__}()',
            job=FunctionJob(
                callable=download_content,
                kwargs=dict(ids=ids, query=query.serialize()),
                args=[],
            ),
        )
        if not self.show('watcher'):
            return G

        for identifier in self.show('watcher'):
            info = self.metadata.get_component('watcher', identifier)
            query = info['dict']['select']
            model, key = identifier.split('/')
            G.add_node(
                f'{model}.predict({key})',
                ComponentJob(
                    component_identifier=model,
                    args=[key],
                    kwargs={
                        'ids': ids,
                        'select': query,
                        **info['dict']['predict_kwargs'],
                    },
                    method_name='predict',
                    variety='model',
                ),
            )

        for identifier in self.show('watcher'):
            model, key = identifier.split('/')
            G.add_edge(
                f'{download_content.__name__}()',
                f'{model}.predict({key})',
            )
            deps = self._get_dependencies_for_watcher(
                identifier
            )  # TODO remove features as explicit argument to watcher
            for dep in deps:
                dep_model, dep_key = dep.split('/')
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
        logging.info('finding documents under filter')
        features = features or {}
        model_identifier = model_info['identifier']
        if features is None:
            features = {}  # pragma: no cover
        documents = list(self.execute(select.select_using_ids(_ids)))
        logging.info('done.')
        documents = [x.unpack() for x in documents]
        if key != '_base' or '_base' in features:
            passed_docs = [r[key] for r in documents]
        else:  # pragma: no cover
            passed_docs = documents
        if model is None:
            model = self.models[model_identifier]
        return model.predict(passed_docs, **(predict_kwargs or {}))

    def _save_artifacts(self, artifact_dictionary: t.Dict):
        raise NotImplementedError  # TODO use in the server code...

    def _add(
        self,
        object: Component,
        dependencies: t.Sequence[Union[Job, str]] = (),
        serialized: t.Optional[t.Dict] = None,
        parent: t.Optional[str] = None,
    ):
        object._on_create(self)

        existing_versions = self.show(object.variety, object.identifier)
        if isinstance(object.version, int) and object.version in existing_versions:
            logging.warn(f'{object.unique_id} already exists - doing nothing')
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
            replace_artifacts(serialized, artifact_info)

        else:
            serialized['version'] = object.version
            serialized['dict']['version'] = object.version

        self._create_children(object, serialized)

        self.metadata.create_component(serialized)
        if parent is not None:
            self.metadata.create_parent_child(parent, object.unique_id)
        object._on_load(self)
        return object.schedule_jobs(self, dependencies=dependencies)

    def _create_children(self, object: Component, serialized: t.Dict):
        for k, child_variety in object.child_components:
            child = getattr(object, k)
            if isinstance(child, str):
                serialized['dict'][k] = {
                    'variety': child_variety,
                    'identifier': child,
                }
                serialized['dict'][k]['version'] = self.metadata.get_latest_version(
                    child_variety, child
                )
            else:
                self._add(
                    child,
                    serialized=serialized['dict'][k],
                    parent=object.unique_id,
                )
                serialized['dict'][k] = {
                    'variety': child.variety,
                    'identifier': child.identifier,
                    'version': child.version,
                }

    def _create_plan(self):
        G = networkx.DiGraph()
        for identifier in self.metadata.show_components('watcher', active=True):
            G.add_node(('watcher', identifier))
        for identifier in self.metadata.show_components('watcher'):
            deps = self._get_dependencies_for_watcher(identifier)
            for dep in deps:
                G.add_edge(('watcher', dep), ('watcher', identifier))
        if not networkx.is_directed_acyclic_graph(G):
            raise ValueError('G is not a directed, acyclic graph')
        return G

    def _delete(self, delete: Delete):
        return self.db.delete(delete)

    def _remove_component_version(
        self,
        variety: str,
        identifier: str,
        version: int,
        force: bool = False,
    ):
        unique_id = Component.make_unique_id(variety, identifier, version)
        if self.metadata.component_version_has_parents(variety, identifier, version):
            parents = self.metadata.get_component_version_parents(unique_id)
            raise Exception(f'{unique_id} is involved in other components: {parents}')

        if force or click.confirm(
            f'You are about to delete {unique_id}, are you sure?',
            default=False,
        ):
            component = self.load(variety, identifier, version=version)
            info = self.metadata.get_component(variety, identifier, version=version)
            if hasattr(component, 'cleanup'):
                component.cleanup(self)
            if variety in self.variety_to_cache_mapping:
                try:
                    del getattr(self, self.variety_to_cache_mapping[variety])[
                        identifier
                    ]
                except KeyError:
                    pass

            if hasattr(component, 'artifacts'):
                for a in component.artifacts:
                    self.artifact_store.delete_artifact(info['dict'][a]['file_id'])
            self.metadata.delete_component_version(variety, identifier, version=version)

    def _download_content(
        self,
        query: t.Optional[Union[Select, Insert]] = None,
        ids=None,
        documents=None,
        timeout=None,
        raises=True,
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
                cursor = self.select(select).raw_cursor  # type: ignore[attr-defined]
                documents = [Document(x) for x in cursor]
        elif isinstance(query, Insert):
            documents = query.documents
        else:
            raise NotImplementedError

        documents = [x.content for x in documents]
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
                documents[id_], key, downloader.results[id_]
            )
        return documents

    def _get_content_for_filter(self, filter) -> Document:
        if isinstance(filter, dict):
            filter = Document(content=filter)  # type: ignore[arg-type]
        if '_id' not in filter.content:
            filter['_id'] = 0
        uris = gather_uris([filter.content])[0]
        if uris:
            output = self._download_content(
                query=None, documents=[filter.content], timeout=None, raises=True
            )[0]
            filter = Document(Document.decode(output, encoders=self.encoders))
        return filter

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

    def _get_object_info(self, identifier, variety, version=None):
        return self.metadata.get_component(variety, identifier, version=version)

    def _apply_watcher(  # noqa: F811
        self,
        identifier,
        ids: Optional[List[str]] = None,
        verbose=False,
        max_chunk_size=5000,
        model=None,
        recompute=False,
        watcher_info=None,
        **kwargs,
    ) -> t.List:
        if watcher_info is None:
            watcher_info = self.metadata.get_component('watcher', identifier)

        select = Serializable.deserialize(watcher_info['select'])
        if ids is None:
            ids = select.get_ids(self)
        else:
            ids = select.select_using_ids(ids=ids).get_ids(self)

        ids = [str(id) for id in ids]  # type: ignore[union-attr]

        if max_chunk_size is not None:
            for it, i in enumerate(range(0, len(ids), max_chunk_size)):
                logging.info(
                    'computing chunk '
                    f'({it + 1}/{math.ceil(len(ids) / max_chunk_size)})'
                )
                self._apply_watcher(
                    identifier,
                    ids=ids[i : i + max_chunk_size],
                    verbose=verbose,
                    max_chunk_size=None,
                    model=model,
                    recompute=recompute,
                    watcher_info=watcher_info,
                    distributed=False,
                    **kwargs,
                )
            return []

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
        type = model_info.get('type')
        if type is not None:
            type = self.encoders[type]
            outputs = [type(x).encode() for x in outputs]

        select.model_update(
            db=self,
            model=watcher_info['model'],
            key=watcher_info['key'],
            outputs=outputs,
            ids=ids,
        )
        return outputs

    # TODO generalize to components
    def replace(
        self,
        object: t.Any,
        upsert: bool = False,
    ):
        """
        (Use-with caution!!) Replace a model in artifact store with updated object.
        :param identifier: model-identifier
        :param object: object to replace
        :param upsert: toggle to ``True`` to enable even if object doesn't exist yet
        """
        try:
            info = self.metadata.get_component(
                object.variety, object.identifier, version=object.version
            )
        except FileNotFoundError as e:
            if upsert:
                return self.add(
                    object,
                )
            raise e

        new_info = object.serialize()
        all_artifacts = list(set(get_artifacts(new_info)))
        artifact_details = dict()
        for a in all_artifacts:
            artifact_details[hash(a)] = a.save(self.artifact_store)
        old_artifacts = list(set(infer_artifacts(info)))
        for oa in old_artifacts:
            self.artifact_store.delete_artifact(oa)

        def replace_artifacts(new_info):
            if isinstance(new_info, dict):
                for k, v in new_info.items():
                    if isinstance(v, Artifact):
                        new_info[k] = artifact_details[hash(v)]
                    else:
                        new_info[k] = replace_artifacts(new_info[k])
            elif isinstance(new_info, list):
                for i, x in enumerate(new_info):
                    if isinstance(x, Artifact):
                        new_info[i] = artifact_details[hash(x)]
                    else:
                        new_info[i] = replace_artifacts(x)
            return new_info

        new_info = replace_artifacts(new_info)

        self.metadata.replace_object(
            new_info,
            identifier=object.identifier,
            variety='model',
            version=object.version,
        )

    def _select_nearest(
        self,
        like: Document,
        vector_index: str,
        ids: Optional[List[str]] = None,
        outputs: Optional[Document] = None,
        n: int = 100,
    ) -> Tuple[List[str], List[float]]:
        like = self._get_content_for_filter(like)  # type: ignore[assignment]
        vector_index = self.vector_indices[vector_index]  # type: ignore[no-redef]

        if outputs is None:
            outs = {}
        else:
            outs = outputs.encode()
            if not isinstance(outs, dict):
                raise TypeError(f'Expected dict, got {type(outputs)}')
        # ruff: noqa: E501
        return vector_index.get_nearest(like, db=self, ids=ids, n=n, outputs=outs)  # type: ignore[attr-defined]


@dc.dataclass
class LoadDict(dict):
    database: Datalayer
    field: str

    def __missing__(self, key: str):
        value = self[key] = self.database.load(self.field, key)
        return value
