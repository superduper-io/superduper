import dataclasses as dc
import math
import typing as t
import warnings
from collections import defaultdict
from typing import Union, Optional, Dict, List, Tuple

import click
import networkx

from superduperdb import CFG
from superduperdb.core.task_workflow import TaskWorkflow
from superduperdb.core.base import Component, strip
from superduperdb.core.documents import Document
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
from superduperdb.fetchers.downloads import Downloader
from superduperdb.fetchers.downloads import gather_uris
from superduperdb.misc.logger import logging
from superduperdb.queries.serialization import from_dict, to_dict
from superduperdb.vector_search.base import VectorDatabase
from superduperdb.core.components import components

# TODO:
# This global variable is a temporary solution to make VectorDatabase available
# to the rest of the code.
# It should be moved to the Server's initialization code where it can be available to
# all threads.
from ...core.fit import Fit
from ...core.job import FunctionJob, ComponentJob, Job

VECTOR_DATABASE = VectorDatabase.create(config=CFG.vector_search)
VECTOR_DATABASE.init().__enter__()

DBResult = t.Any
TaskGraph = t.Any

DeleteResult = DBResult
InsertResult = t.Tuple[DBResult, t.Optional[TaskGraph]]
SelectResult = t.List[Document]
UpdateResult = t.Any

ExecuteQuery = t.Union[Select, Delete, Update, Insert]
ExecuteResult = t.Union[SelectResult, DeleteResult, UpdateResult, InsertResult]


class BaseDatabase:
    """
    Base database connector for SuperDuperDB - all database types should subclass this
    type.
    """

    _database_type: str
    name: str
    select_cls: t.Type[Select]
    models: t.Dict

    variety_to_cache_mapping = {
        'model': 'models',
        'metric': 'metrics',
        'type': 'types',
        'vector_index': 'vector_indices',
    }

    def __init__(
        self,
        databackend: BaseDataBackend,
        metadata: MetaDataStore,
        artifact_store: ArtifactStore,
    ):
        self.metrics = LoadDict(self, 'metric')
        self.models = LoadDict(self, 'model')
        self.types = LoadDict(self, 'type')
        self.vector_indices = LoadDict(self, 'vector_index')

        self.remote = CFG.remote
        self.metadata = metadata
        self.artifact_store = artifact_store
        self.databackend = databackend

    @property
    def db(self):
        return self.databackend.db

    def validate(
        self,
        identifier: str,
        variety: str,
        validation_datasets: List[str],
        metrics: List[str],
    ):
        """
        Evaluate quality of component, using `Component.validate`, if implemented.

        :param identifier: identifier of semantic index
        :param variety: variety of component
        :param validation_datasets: validation-sets on which to validate
        :param metrics: metric functions to compute
        """
        component = self.load(variety, identifier)
        metrics = [self.load('metric', m) for m in metrics]  # type: ignore[misc]
        for vs in validation_datasets:
            res = component.validate(self, vs, metrics)  # type: ignore[attr-defined]
            for m in res:
                self.metadata.update_object(
                    identifier,
                    variety,
                    f'final_metrics.{vs}.{m}',
                    res[m],
                )

    def show(
        self,
        variety: str,
        identifier: Optional[str] = None,
        version: Optional[int] = None,
    ):
        """
        Show available functionality which has been added using ``self.add``.
        If version is specified, then print full metadata

        :param variety: variety of component to show ["type", "model", "watcher",
                       "learning_task", "training_configuration", "metric",
                       "vector_index", "job"]
        :param identifier: identifying string to component
        :param version: (optional) numerical version - specify for full metadata
        """
        if identifier is None:
            assert version is None, f"must specify {identifier} to go with {version}"
            return self.metadata.show_components(variety=variety)
        elif identifier is not None and version is None:
            return self.metadata.show_component_versions(
                variety=variety, identifier=identifier
            )
        elif identifier is not None and version is not None:
            if version == -1:
                return self._get_object_info(variety=variety, identifier=identifier)
            else:
                return self._get_object_info(
                    variety=variety, identifier=identifier, version=version
                )
        else:
            raise ValueError(
                f'Incorrect combination of {variety}, {identifier}, {version}'
            )

    def predict(
        self,
        model_identifier: str,
        input: Document,
    ) -> Union[List[Document], Document]:
        """
        Apply model to input.

        :param model_identifier: model or ``str`` referring to an uploaded model
        :param input: input to be passed to the model.
                      Must be possible to encode with registered types
        """
        model = self.models[model_identifier]
        opts = self.metadata.get_component('model', model_identifier)
        out = model.predict(input.unpack(), **opts.get('predict_kwargs', {}))
        if model.encoder is not None:
            out = model.encoder(out)  # type: ignore
        return Document(out)

    def execute(self, query: ExecuteQuery) -> ExecuteResult:
        """
        Execute a query on the datalayer

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
        return delete(self)

    def insert(self, insert: Insert) -> InsertResult:
        return insert(self)

    def run(
        self,
        job,
        depends_on: t.Optional[t.List] = None,
        remote: t.Optional[bool] = None,
    ):
        if remote is None:
            remote = CFG.remote
        return job(db=self, dependencies=depends_on, remote=remote)

    def select(self, select: Select) -> SelectResult:
        return select(self)

    def like(self, like: Like) -> SelectResult:
        return like(self)

    def select_one(self, select_one: SelectOne) -> SelectResult:
        return select_one(self)

    def refresh_after_update_or_insert(
        self, query: Union[Select, Update], ids: List[str], verbose=False
    ):
        task_graph: TaskWorkflow = self._build_task_workflow(
            query.select_table,
            ids=ids,
            verbose=verbose,
        )
        task_graph(db=self)
        return task_graph

    def update(self, update: Update) -> UpdateResult:
        return update(self)

    def add(
        self,
        object: Component,
        serializer: str = 'dill',
        serializer_kwargs: Optional[Dict] = None,
        dependencies: t.Sequence[Union[Job, str]] = (),
    ):
        """
        Add functionality in the form of components. Components are stored in the
        configured artifact store, and linked to the primary datalayer through
        the metadata.

        :param object: Object to be stored
        :param serializer: Serializer to use to convert component to ``bytes``
        :param serializer_kwargs: kwargs to be passed to ``serializer``
        """
        return self._add(
            object=object,
            serializer=serializer,
            serializer_kwargs=serializer_kwargs,
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

        :param variety: variety of component to remove ["type", "model", "watcher",
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
        repopulate: bool = True,
        allow_hidden: bool = False,
    ) -> Component:
        """
        Load component using uniquely identifying information.

        :param variety: variety of component to remove ["type", "model", "watcher",
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
        if 'serializer' not in info:
            info['serializer'] = 'dill'
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

        G.add_node(
            f'{download_content.__name__}()',
            job=FunctionJob(
                callable=download_content,
                kwargs=dict(ids=ids, query=to_dict(query)),
                args=[],
            ),
        )
        if not self.show('watcher'):
            return G

        for identifier in self.show('watcher'):
            if isinstance(query, Update):
                query = query.select
            model, key = identifier.split('/')
            G.add_node(
                f'{model}.predict({key})',
                ComponentJob(
                    component_identifier=model,
                    args=[key],
                    kwargs={'ids': ids, 'select': to_dict(query)},
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

    def _add(
        self,
        object: Component,
        serializer: str = 'dill',
        serializer_kwargs: Optional[Dict] = None,
        parent: Optional[str] = None,
        dependencies: t.Sequence[Union[Job, str]] = (),
    ):
        if object.repopulate_on_init:
            object.repopulate(self)

        existing_versions = self.show(object.variety, object.identifier)
        if isinstance(object.version, int) and object.version in existing_versions:
            logging.warn(f'{object.unique_id} already exists - doing nothing')
            return
        version = existing_versions[-1] + 1 if existing_versions else 0
        object.version = version

        for c in object.child_components:
            logging.info(f'Checking upstream-component {c.variety}/{c.identifier}')
            self._add(
                c,
                serializer=serializer,
                serializer_kwargs=serializer_kwargs,
                parent=object.unique_id,
                dependencies=dependencies,
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
                'serializer': serializer,
                'serializer_kwargs': serializer_kwargs or {},
            }
        )
        if parent is not None:
            self.metadata.create_parent_child(parent, object.unique_id)
        logging.info(f'Created {object.unique_id}')

        object.repopulate(self)
        return object.schedule_jobs(self, dependencies=dependencies)

    def _create_plan(self):
        G = networkx.DiGraph()
        for identifier in self.metadata.show_components('watcher', active=True):
            G.add_node(('watcher', identifier))
        for identifier in self.metadata.show_components('watcher'):
            deps = self._get_dependencies_for_watcher(identifier)
            for dep in deps:
                G.add_edge(('watcher', dep), ('watcher', identifier))
        assert networkx.is_directed_acyclic_graph(G)
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
            info = self.metadata.get_component(variety, identifier, version=version)
            component_cls = components[variety]
            if hasattr(component_cls, 'cleanup'):
                component_cls.cleanup(info, self)
            if variety in self.variety_to_cache_mapping:
                try:
                    del getattr(self, self.variety_to_cache_mapping[variety])[
                        identifier
                    ]
                except KeyError:
                    pass
            self.artifact_store.delete_artifact(info['object'])
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
            filter = Document(filter)
        if '_id' not in filter.content:
            filter['_id'] = 0
        uris = gather_uris([filter.content])[0]
        if uris:
            output = self._download_content(
                query=None, documents=[filter.content], timeout=None, raises=True
            )[0]
            filter = Document(Document.decode(output, types=self.types))
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

        select = from_dict(watcher_info['select'])
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
                    remote=False,
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
            type = self.types[type]
            outputs = [type(x).encode() for x in outputs]

        select.model_update(
            db=self,
            model=watcher_info['model'],
            key=watcher_info['key'],
            outputs=outputs,
            ids=ids,
        )
        return outputs

    def replace_model(
        self,
        identifier: str,
        object: t.Any,
        upsert: bool = False,
    ):
        try:
            info = self.metadata.get_component(
                'model', identifier, version=object.version
            )
        except FileNotFoundError as e:
            if upsert:
                return self.add(
                    object,
                )
            raise e

        file_id, _ = self.artifact_store.create_artifact(
            object,
            serializer=info['serializer'],
            serializer_kwargs=info['serializer_kwargs'],
        )
        self.artifact_store.delete_artifact(info['object'])
        self.metadata.update_object(
            identifier=identifier,
            variety='model',
            key='object',
            value=file_id,
            version=object.version,
        )

    def _select_nearest(
        self,
        like: t.Dict,
        vector_index: str,
        ids: Optional[List[str]] = None,
        outputs: Optional[Document] = None,
        n: int = 100,
    ) -> Tuple[List[str], List[float]]:
        like = Document(like)  # type: ignore[assignment]
        like = self._get_content_for_filter(like)  # type: ignore[assignment]
        vector_index = self.vector_indices[vector_index]  # type: ignore[no-redef]

        if outputs is None:
            outs = {}
        else:
            outs = outputs.encode()
            if not isinstance(outs, dict):
                raise TypeError(f'Expected dict, got {type(outputs)}')
        # ruff: noqa: E501
        return vector_index.get_nearest(like, database=self, ids=ids, n=n, outputs=outs)  # type: ignore[attr-defined]

    def _fit(self, identifier: str) -> None:
        """
        Execute the learning task.

        :param identifier: Identifier of a learning task.
        """

        fit: Fit = self.load('fit', identifier)  # type: ignore
        # ruff: noqa: E501
        try:
            fit.model.fit(  # type: ignore[attr-defined]
                *fit.keys,  # type: ignore[attr-defined]
                db=self,
                select=fit.select,  # type: ignore[attr-defined]
                validation_sets=fit.validation_sets,  # type: ignore[attr-defined]
                metrics=fit.metrics,  # type: ignore[attr-defined]
                configuration=fit.training_configuration,  # type: ignore[attr-defined]
            )
        except Exception as e:
            self.remove('fit', identifier, force=True)
            raise e


@dc.dataclass
class LoadDict(dict):
    database: BaseDatabase
    field: str

    def __missing__(self, key: str):
        value = self[key] = self.database.load(self.field, key)
        return value
