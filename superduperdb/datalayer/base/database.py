import warnings
from collections import defaultdict
import math
import random
from typing import Union, Optional, Dict, List, Tuple

import click
import networkx

from superduperdb import CFG
from superduperdb.cluster.job_submission import work
from superduperdb.cluster.task_workflow import TaskWorkflow
from superduperdb.core.base import Component, strip
from superduperdb.core.documents import Document
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
from superduperdb.vector_search.base import VectorDatabase


# TODO:
# This global variable is a temporary solution to make VectorDatabase available
# to the rest of the code.
# It should be moved to the Server's initialization code where it can be available to
# all threads.
VECTOR_DATABASE = VectorDatabase.create(config=CFG.vector_search)
VECTOR_DATABASE.init().__enter__()


class BaseDatabase:
    """
    Base database connector for SuperDuperDB - all database types should subclass this
    type.
    """

    select_cls = Select
    type_id_to_cache_mapping = {
        'model': 'models',
        'metric': 'metrics',
        'type': 'types',
        'vector_index': 'vector_indices',
    }

    def __init__(self, metadata: MetaDataStore, artifact_store: ArtifactStore):
        self.metrics = ArgumentDefaultDict(lambda x: self.load('metric', x))
        self.models = ArgumentDefaultDict(lambda x: self.load('model', x))
        self.types = ArgumentDefaultDict(lambda x: self.load('type', x))
        self.vector_indices = ArgumentDefaultDict(
            lambda x: self.load('vector_index', x)
        )

        self.remote = CFG.remote
        self.metadata = metadata
        self.artifact_store = artifact_store

    @work
    def validate(
        self,
        identifier: str,
        type_id: str,
        validation_sets: List[str],
        metrics: List[str],
    ):
        """
        Evaluate quality of component, using `Component.validate`, if implemented.

        :param identifier: identifier of semantic index
        :param type_id: type_id of component
        :param validation_sets: validation-sets on which to validate
        :param metrics: metric functions to compute
        """
        component = self.load(type_id, identifier)
        metrics = [self.load('metric', m) for m in metrics]
        validation_selects = [
            self.get_query_for_validation_set(vs) for vs in validation_sets
        ]
        results = component.validate(self, validation_selects, metrics)
        for vs, res in zip(validation_sets, results):
            for m in res:
                self.metadata.update_object(
                    identifier,
                    type_id,
                    f'final_metrics.{vs}.{m}',
                    res[m],
                )

    def show(
        self,
        type_id: str,
        identifier: Optional[str] = None,
        version: Optional[int] = None,
    ):
        """
        Show available functionality which has been added using ``self.add``.
        If version is specified, then print full metadata

        :param type_id: type_id of component to show
        :param identifier: identifying string to component
        :param version: (optional) numerical version - specify for full metadata
        """
        if identifier is None:
            assert version is None, f"must specify {identifier} to go with {version}"
            return self._show_components(type_id)
        elif identifier is not None and version is None:
            return self._show_component_versions(type_id, identifier)
        elif identifier is not None and version is not None:
            if version == -1:
                return self._get_object_info(type_id, identifier)
            else:
                return self._get_object_info(type_id, identifier, version)
        else:
            raise ValueError(
                f'Incorrect combination of {type_id}, {identifier}, {version}'
            )

    def predict(
        self,
        model_identifier: str,
        input: Union[List[Document], Document],
    ) -> Union[List[Document], Document]:
        """
        Apply model to input.

        :param model_identifier: model or ``str`` referring to an uploaded model
        :param input: input to be passed to the model.
                      Must be possible to encode with registered types
        :param kwargs: key-values (see ``superduperdb.models.utils.predict``)
        """
        model: Model = self.models[model_identifier]
        opts = self.metadata.get_component('model', model_identifier)
        if isinstance(input, Document):
            out = model.predict_one(input.unpack(), **opts.get('predict_kwargs', {}))
            if model.type is not None:
                out = model.type(out)
            return Document(out)

        out = model.predict(
            [x.unpack() for x in input], **opts.get('predict_kwargs', {})
        )
        to_return = []
        for x in out:
            if model.type is not None:
                x = model.type(x)
            to_return.append(Document(x))
        return to_return

    def execute(
        self,
        query: Union[Select, Delete, Update, Insert],
        refresh: bool = True,
        raw: bool = False,
        verbose: bool = True,
    ):
        """
        Execute a query on the datalayer

        :param query: select, insert, delete, update,
        :param refresh: refresh the computations if applicable
        :param raw: toggle to ``True`` to return raw data, including
                    ``bytes`` from data base
        :param verbose: toggle to ``False`` to suppress output
        """
        if isinstance(query, Select):
            return self._select(query, raw=raw)
        elif isinstance(query, Delete):
            return self._delete(query)
        elif isinstance(query, Update):
            return self._update(query, refresh=refresh, verbose=verbose)
        elif isinstance(query, Insert):
            return self._insert(query, refresh=refresh, verbose=verbose)
        else:
            raise TypeError(
                f'Wrong type of {query}; '
                f'Expected object of type {Union[Select, Delete, Update, Insert]}; '
                f'Got {type(query)};'
            )

    def add(
        self,
        object: Component,
        serializer: str = 'pickle',
        serializer_kwargs: Optional[Dict] = None,
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
        )

    def remove(
        self,
        type_id: str,
        identifier: str,
        version: Optional[int] = None,
        force=False,
    ):
        """
        Remove component (version: optional)

        :param type_id: type_id of component to remove ["type", "model", "watcher",
                        "training_configuration", "learning_task", "vector_index"]
        :param identifier: identifier of component (see `core.base.Component`)
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
        version: Optional[int] = None,
        repopulate: bool = True,
        allow_hidden: bool = False,
    ) -> Component:
        """
        Load component using uniquely identifying information.

        :param type_id: type_id of component to remove ["type", "model", "watcher",
                        "training_configuration", "learning_task", "vector_index"]
        :param identifier: identifier of component (see `core.base.Component`)
        :param version: [optional] numerical version
        :param repopulate: toggle to ``False`` to only load references to other
                           components
        :param allow_hidden: toggle to ``True`` to allow loading of deprecated
                             components
        """
        info = self.metadata.get_component(
            type_id, identifier, version=version, allow_hidden=allow_hidden
        )
        if info is None:
            raise Exception(
                f'No such object of type "{type_id}", '
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
        if cm := self.type_id_to_cache_mapping.get(type_id):
            getattr(self, cm)[m.identifier] = m
        return m

    def _build_task_workflow(
        self, select: Select, ids=None, dependencies=(), verbose=True
    ):
        job_ids = defaultdict(lambda: [])
        job_ids.update(dependencies)
        G = TaskWorkflow(self)
        if ids is None:
            ids = self._get_ids_from_select(select.select_only_id)

        G.add_node(
            f'{self._download_content.__name__}()',
            data={
                'task': self._download_content,
                'args': [
                    select,
                ],
                'kwargs': {
                    'ids': ids,
                },
            },
        )
        if not self._show_components('watcher'):
            return G

        for identifier in self._show_components('watcher'):
            G.add_node(
                f'{self._apply_watcher.__name__}({identifier})',
                data={
                    'task': self._apply_watcher,
                    'args': [identifier],
                    'kwargs': {
                        'ids': ids,
                        'verbose': verbose,
                    },
                },
            )

        for identifier in self._show_components('watcher'):
            G.add_edge(
                f'{self._download_content.__name__}()',
                f'{self._apply_watcher.__name__}({identifier})',
            )
            deps = self._get_dependencies_for_watcher(identifier)
            for dep in deps:
                G.add_edge(
                    f'{self._apply_watcher.__name__}({dep})',
                    f'{self._apply_watcher.__name__}({identifier})',
                )
                G.add_edge(
                    f'{self._download_content.__name__}()',
                    f'{self._apply_watcher.__name__}({identifier})',
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
        documents = list(self._select(select.select_using_ids(_ids, features=features)))
        logging.info('done.')
        documents = [x.unpack() for x in documents]
        if key != '_base' or '_base' in features:
            passed_docs = [r[key] for r in documents]
        else:  # pragma: no cover
            passed_docs = documents
        if model is None:
            model = self.models[model_identifier]
        return model.predict(passed_docs, **(predict_kwargs or {}))

    def _create_job_record(self, *args, **kwargs):  # TODO - move to metadata
        raise NotImplementedError

    def _add_split_to_row(self, r, other):
        raise NotImplementedError

    def _base_insert(self, insert: Insert):
        raise NotImplementedError

    def _add(
        self,
        object: Component,
        serializer: str = 'pickle',
        serializer_kwargs: Optional[Dict] = None,
        parent: Optional[str] = None,
    ):
        existing_versions = self._show_component_versions(
            object.type_id, object.identifier
        )
        if isinstance(object.version, int) and object.version in existing_versions:
            logging.warn(f'{object.unique_id} already exists - doing nothing')
            return
        version = existing_versions[-1] + 1 if existing_versions else 0
        object.version = version

        for c in object.child_components:
            logging.info(f'Checking upstream-component {c.type_id}/{c.identifier}')
            self._add(
                c,
                serializer=serializer,
                serializer_kwargs=serializer_kwargs,
                parent=object.unique_id,
            )

        for p in object.child_references:
            if p.version is None:
                p.version = self.metadata.get_latest_version(p.type_id, p.identifier)

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
                'type_id': object.type_id,
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
        for identifier in self.metadata.show_components('watcher', active=True):
            G.add_node(('watcher', identifier))
        for identifier in self.metadata.show_components('watcher'):
            deps = self._get_dependencies_for_watcher(identifier)
            for dep in deps:
                G.add_edge(('watcher', dep), ('watcher', identifier))
        assert networkx.is_directed_acyclic_graph(G)
        return G

    def _add_validation_set(self, identifier, select: Select, chunk_size=1000):
        if identifier in self._show_validation_sets():
            raise Exception(f'validation set {identifier} already exists!')

        data = self._select(select)
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

    def _delete(self, delete: Delete):
        return self._base_delete(delete)

    def _remove_component_version(
        self,
        type_id: str,
        identifier: str,
        version: int,
        force: bool = False,
    ):
        unique_id = Component.make_unique_id(type_id, identifier, version)
        if self.metadata.component_version_has_parents(type_id, identifier, version):
            parents = self.metadata.get_component_version_parents(
                type_id, identifier, version
            )
            raise Exception(f'{unique_id} is involved in other components: {parents}')

        if force or click.confirm(
            f'You are about to delete {unique_id}, are you sure?',
            default=False,
        ):
            info = self.metadata.get_component(type_id, identifier, version=version)
            if type_id in self.type_id_to_cache_mapping:
                try:
                    del getattr(self, self.type_id_to_cache_mapping[type_id])[
                        identifier
                    ]
                except KeyError:
                    pass
            self.artifact_store.delete_artifact(info['object'])
            self.metadata.delete_component_version(type_id, identifier, version=version)

    @work
    def _download_content(
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
                documents = list(self._select(query))
            else:
                documents = list(self._select(query.select_using_ids(ids), raw=True))
                documents = [Document(x) for x in documents]
        else:
            documents = query.documents

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

        def update_one(id, key, bytes):
            return self._update(self._download_update(query.table, id, key, bytes))

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
            filter = self._download_content(
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

    def _get_output_from_document(self, r: Document, key: str, model: str):
        raise NotImplementedError

    def _get_job_info(self, identifier):
        raise NotImplementedError

    def _get_ids_from_select(self, select: Select):
        raise NotImplementedError

    def _get_raw_cursor(self, select: Select):
        raise NotImplementedError

    def _get_object_info(self, identifier, type_id, version=None):
        return self.metadata.get_component(type_id, identifier, version=version)

    def get_query_for_validation_set(self, validation_set):
        raise NotImplementedError

    def _get_watcher_for_learning_task(self, learning_task):
        info = self.metadata.get_component('learning_task', learning_task)
        key_to_watch = info['keys_to_watch'][0]
        model_identifier = next(
            m for i, m in enumerate(info['models']) if info['keys'][i] == key_to_watch
        )
        return f'[{learning_task}]:{model_identifier}/{key_to_watch}'

    def _insert(self, insert: Insert, refresh=True, verbose=True):
        for item in insert.documents:
            r = random.random()
            try:
                valid_probability = self.metadata.get_metadata(key='valid_probability')
            except TypeError:
                valid_probability = 0.05  # TODO proper error handling
            if '_fold' not in item.content:
                item['_fold'] = 'valid' if r < valid_probability else 'train'
        output = self._base_insert(insert)
        if not refresh:  # pragma: no cover
            return output, None
        task_graph = self._build_task_workflow(
            insert.select_table, ids=output.inserted_ids, verbose=verbose
        )
        task_graph()
        return output, task_graph

    def _insert_validation_data(self, tmp, identifier):
        raise NotImplementedError

    def _show_jobs(self):
        """
        List jobs
        """
        return self.metadata.show_jobs()

    def _show_components(self, type_id):
        return self.metadata.show_components(type_id)

    def _show_component_versions(self, type_id: str, identifier: str):
        return sorted(self.metadata.show_component_versions(type_id, identifier))

    def _show_validation_sets(self):
        """
        List validation sets.
        """
        return self.metadata.show_components(type_id='validation_set')

    @work
    def _apply_watcher(  # noqa: F811
        self,
        identifier,
        ids: List[str] = None,
        verbose=False,
        max_chunk_size=5000,
        model=None,
        recompute=False,
        watcher_info=None,
        **kwargs,
    ):
        if watcher_info is None:
            watcher_info = self.metadata.get_component('watcher', identifier)
        select = self.select_cls(**watcher_info['_select'])
        if ids is None:
            ids = self._get_ids_from_select(select.select_only_id)
            ids = [str(id) for id in ids]
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
        type = model_info.get('type')
        if type is not None:
            type = self.types[type]
            outputs = [type(x).encode() for x in outputs]
        self._write_watcher_outputs(watcher_info, outputs, ids)
        return outputs

    def _replace_model(self, identifier: str, object: Model):
        info = self.metadata.get_component('model', identifier, version=object.version)
        if 'serializer' not in info:
            info['serializer'] = 'pickle'
        if 'serializer_kwargs' not in info:
            info['serializer_kwargs'] = {}
        assert identifier in self.metadata.show_components(
            'model'
        ), f'model "{identifier}" doesn\'t exist to replace'
        assert object.version in self.metadata.show_component_versions(
            'model', identifier
        )

        file_id = self.artifact_store.create_artifact(
            object,
            serializer=info['serializer'],
            serializer_kwargs=info['serializer_kwargs'],
        )
        self.artifact_store.delete_artifact(info['object'])
        self.metadata.update_object(identifier, 'model', 'object', file_id)

    def _select(self, select: Select, raw: bool = False) -> List[Document]:
        if select.like is not None:
            if select.similar_first:
                return self._select_similar_then_matches(select, raw=raw)
            else:
                return self._select_matches_then_similar(select, raw=raw)
        else:
            if raw:
                return self._get_raw_cursor(select)
            else:
                return self._get_cursor(select, features=select.features)

    def _select_matches_then_similar(self, select: Select, raw: bool = False):
        if not select.is_trivial:
            id_cursor = self._get_raw_cursor(select.select_only_id)
            ids = [x['_id'] for x in id_cursor]
            similar_ids, scores = self._select_nearest(select, ids=ids)
        else:
            similar_ids, scores = self._select_nearest(select)

        if raw:
            return self._get_raw_cursor(select.select_using_ids(similar_ids))
        else:
            return self._get_cursor(
                select.select_using_ids(similar_ids),
                features=select.features,
                scores=dict(zip(similar_ids, scores)),
            )

    def _select_similar_then_matches(self, select: Select, raw: bool = False):
        similar_ids, scores = self._select_nearest(select)

        if raw:
            return self._get_raw_cursor(select.select_using_ids(similar_ids))
        else:
            return self._get_cursor(
                select.select_using_ids(similar_ids),
                features=select.features,
                scores=dict(zip(similar_ids, scores)),
            )

    def _select_nearest(
        self, select: Select, ids: Optional[List[str]] = None
    ) -> Tuple[List[str], List[float]]:
        if select.download and select.like is not None:
            like = self._get_content_for_filter(select.like)  # pragma: no cover
        else:
            like = select.like

        vector_index: VectorIndex = self.vector_indices[select.vector_index]
        return vector_index.get_nearest(
            like, database=self, ids=ids, n=select.n, outputs=select.outputs
        )

    def _separate_query_part_from_validation_record(self, r):
        """
        Separate the info in the record after splitting.

        :param r: record
        """
        raise NotImplementedError

    def _set_content_bytes(self, r, key, bytes_):
        raise NotImplementedError

    def _set_job_flag(self, identifier, kw):
        """
        Set key-value pair in job record

        :param identifier: id of job
        :param kw: tuple of key-value pair
        """
        raise NotImplementedError

    @work
    def _fit(self, identifier):
        """
        Execute the learning task.

        :param identifier: Identifier of a learning task.
        """

        learning_task: LearningTask = self.load('learning_task', identifier)

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
            self.remove('learning_task', identifier, force=True)
            raise e

    def _update(self, update: Update, refresh=True, verbose=True):
        if refresh and self.metadata.show_components('model'):
            ids = self._get_ids_from_select(update.select_ids)
        result = self._base_update(update)
        if refresh and self.metadata.show_components('model'):
            task_graph = self._build_task_workflow(
                update.select, ids=ids, verbose=verbose
            )
            task_graph()
            return result, task_graph
        return result

    def _watch_job(self, identifier):
        """
        Watch stdout/stderr of worker job.

        :param identifier: job-id
        """
        return self.metadata.watch_job(identifier)

    def _write_output_to_job(self, identifier, msg, stream):
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
