import math
import random
import typing as t
import warnings
from collections import defaultdict
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
from superduperdb.datalayer.base.data_backend import BaseDataBackend
from superduperdb.datalayer.base.metadata import MetaDataStore
from superduperdb.datalayer.base.query import Insert, Select, Delete, Update
from superduperdb.fetchers.downloads import Downloader
from superduperdb.fetchers.downloads import gather_uris
from superduperdb.misc import progress
from superduperdb.misc.logger import logging
from superduperdb.misc.special_dicts import ArgumentDefaultDict
from superduperdb.vector_search.base import VectorDatabase
from superduperdb.core import components

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

    _database_type: str
    name: str
    select_cls: t.Type[Select]
    models: t.Dict[str, Model]

    variety_to_cache_mapping = {
        'model': 'models',
        'metric': 'metrics',
        'type': 'types',
        'vector_index': 'vector_indices',
    }

    def __init__(
        self,
        db: BaseDataBackend,
        metadata: MetaDataStore,
        artifact_store: ArtifactStore,
    ):
        self.metrics = ArgumentDefaultDict(
            lambda x: self.load('metric', x)  # type: ignore
        )
        self.models = ArgumentDefaultDict(
            lambda x: self.load('model', x)  # type: ignore
        )
        self.types = ArgumentDefaultDict(lambda x: self.load('type', x))  # type: ignore
        self.vector_indices = ArgumentDefaultDict(
            lambda x: self.load('vector_index', x)  # type: ignore
        )

        self.remote = CFG.remote
        self.metadata = metadata
        self.artifact_store = artifact_store
        self.db = db

    @work
    def validate(
        self,
        identifier: str,
        variety: str,
        validation_sets: t.List[str],
        metrics: t.List[str],
    ):
        """
        Evaluate quality of component, using `Component.validate`, if implemented.

        :param identifier: identifier of semantic index
        :param variety: variety of component
        :param validation_sets: validation-sets on which to validate
        :param metrics: metric functions to compute
        """
        component = self.load(variety, identifier)
        metrics = [self.load('metric', m) for m in metrics]  # type: ignore
        validation_selects = [
            self.db.get_query_for_validation_set(vs) for vs in validation_sets
        ]

        results = component.validate(self, validation_selects, metrics)  # type: ignore
        for vs, res in zip(validation_sets, results):
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
            return self.metadata.show_components(variety)
        elif identifier is not None and version is None:
            return self.metadata.show_component_versions(variety, identifier)
        elif identifier is not None and version is not None:
            if version == -1:
                return self._get_object_info(variety, identifier)
            else:
                return self._get_object_info(variety, identifier, version)
        else:
            raise ValueError(
                f'Incorrect combination of {variety}, {identifier}, {version}'
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
            if model.encoder is not None:
                out = model.encoder(out)  # type: ignore
            return Document(out)

        out = model.predict(
            [x.unpack() for x in input], **opts.get('predict_kwargs', {})
        )
        to_return = []
        for x in out:
            if model.encoder is not None:
                x = model.encoder(x)  # type: ignore
            to_return.append(Document(x))
        return to_return

    def execute(
        self,
        query: Union[Select, Delete, Update, Insert],
        refresh: bool = True,
        verbose: bool = True,
    ):
        """
        Execute a query on the datalayer

        :param query: select, insert, delete, update,
        :param refresh: refresh the computations if applicable
        :param verbose: toggle to ``False`` to suppress output
        """
        if isinstance(query, Select):
            return self._select(query)
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
    ) -> t.Type[Component]:
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

    def _build_task_workflow(
        self, select: Select, ids=None, dependencies=(), verbose=True
    ):
        job_ids: t.Dict[str, t.Any] = defaultdict(lambda: [])
        job_ids.update(dependencies)
        G = TaskWorkflow(self)
        if ids is None:
            ids = self.db.get_ids_from_select(select.select_only_id)

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
        if not self.show('watcher'):
            return G

        for identifier in self.show('watcher'):
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

        for identifier in self.show('watcher'):
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
        documents = list(self.execute(select.select_using_ids(_ids, features=features)))
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
        for identifier in self.metadata.show_components('watcher', active=True):
            G.add_node(('watcher', identifier))
        for identifier in self.metadata.show_components('watcher'):
            deps = self._get_dependencies_for_watcher(identifier)
            for dep in deps:
                G.add_edge(('watcher', dep), ('watcher', identifier))
        assert networkx.is_directed_acyclic_graph(G)
        return G

    def _add_validation_set(self, identifier, select: Select, chunk_size=1000):
        if identifier in self.db.show_validation_sets():
            raise Exception(f'validation set {identifier} already exists!')

        data = self._select(select)
        it = 0
        tmp = []
        for r in progress.progressbar(data):
            tmp.append(r)
            it += 1
            if it % chunk_size == 0:
                self.db.insert_validation_data(tmp, identifier)
                tmp = []
        if tmp:
            self.db.insert_validation_data(tmp, identifier)

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
                select = query.select_using_ids(ids)
                select = select.copy(update={'raw': True})
                documents = list(self._select(select))
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
            return self._update(self.db.download_update(query.table, id, key, bytes))

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
            documents[id_] = self.db.set_content_bytes(
                documents[id_], key, downloader.results[id_]
            )
        return documents

    def _get_content_for_filter(self, filter):
        if '_id' not in filter:
            filter['_id'] = 0
        uris = gather_uris([filter])[0]
        if uris:
            output = self._download_content(
                self.name, documents=[filter.content], timeout=None, raises=True
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
            if '_fold' not in item.content:  # type: ignore
                item['_fold'] = 'valid' if r < valid_probability else 'train'
        output = self.db.insert(insert)
        if not refresh:  # pragma: no cover
            return output, None
        task_graph = self._build_task_workflow(
            insert.select_table, ids=output.inserted_ids, verbose=verbose
        )
        task_graph()
        return output, task_graph

    @work
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
    ):
        if watcher_info is None:
            watcher_info = self.metadata.get_component('watcher', identifier)
        select = self.db.select_cls(**watcher_info['select'])  # type: ignore
        if ids is None:
            ids = self.db.get_ids_from_select(select.select_only_id)
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
        self.db.write_outputs(watcher_info, outputs, ids)
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

    def _select(self, select: Select) -> List[Document]:
        if select.like is not None:
            if select.similar_first:
                return self._select_similar_then_matches(select)
            else:
                return self._select_matches_then_similar(select)
        else:
            if select.raw:
                return self.db.get_raw_cursor(select)
            else:
                return self.db.get_cursor(
                    select,
                    features=select.features,
                    types=self.types,
                )

    def _select_matches_then_similar(self, select: Select):
        if not select.is_trivial:
            id_cursor = self.db.get_raw_cursor(select.select_only_id)
            ids = [x['_id'] for x in id_cursor]
            similar_ids, scores = self._select_nearest(select, ids=ids)
        else:
            similar_ids, scores = self._select_nearest(select)

        if select.raw:
            return self.db.get_raw_cursor(select.select_using_ids(similar_ids))
        else:
            return self.db.get_cursor(
                select.select_using_ids(similar_ids),
                features=select.features,
                scores=dict(zip(similar_ids, scores)),
                types=self.types,
            )

    def _select_similar_then_matches(self, select: Select):
        similar_ids, scores = self._select_nearest(select)

        if select.raw:
            return self.db.get_raw_cursor(select.select_using_ids(similar_ids))
        else:
            return self.db.get_cursor(
                select.select_using_ids(similar_ids),
                features=select.features,
                scores=dict(zip(similar_ids, scores)),
                types=self.types,
            )

    def _select_nearest(
        self, select: Select, ids: Optional[List[str]] = None
    ) -> Tuple[List[str], List[float]]:
        if select.like is not None:
            like = select.like()
        else:
            raise ValueError('_select_nearest requires non-empty select.like')

        if select.download:
            like = self._get_content_for_filter(like)  # pragma: no cover

        vector_index: VectorIndex = self.vector_indices[select.vector_index]
        if select.outputs is None:
            outputs = {}
        else:
            outputs = select.outputs().encode()
            if not isinstance(outputs, dict):
                raise TypeError(f'Expected dict, got {type(outputs)}')
        return vector_index.get_nearest(
            like, database=self, ids=ids, n=select.n, outputs=outputs
        )

    @work
    def _fit(self, identifier) -> None:
        """
        Execute the learning task.

        :param identifier: Identifier of a learning task.
        """

        learning_task: LearningTask = self.load(
            'learning_task', identifier
        )  # type: ignore
        trainer = learning_task.training_configuration(
            identifier=identifier,
            keys=learning_task.keys,
            model_names=learning_task.models.aslist(),
            models=learning_task.models,
            select=learning_task.select,
            validation_sets=learning_task.validation_sets,
            metrics={m.identifier: m for m in learning_task.metrics},
            features=learning_task.features,
        )  # type: ignore

        try:
            trainer()
        except Exception as e:
            self.remove('learning_task', identifier, force=True)
            raise e

    def _update(self, update: Update, refresh=True, verbose=True) -> t.Any:
        if refresh and self.metadata.show_components('model'):
            ids = self.db.get_ids_from_select(update.select_ids)
        result = self.db.update(update)
        if refresh and self.metadata.show_components('model'):
            task_graph = self._build_task_workflow(
                update.select, ids=ids, verbose=verbose
            )
            task_graph()
            return result, task_graph
        return result
