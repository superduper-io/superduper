import dataclasses as dc
import random
import typing as t
import warnings
from collections import defaultdict, namedtuple

import click
import networkx
import tqdm

import superduperdb as s
from superduperdb import logging
from superduperdb.backends.base.artifacts import ArtifactStore
from superduperdb.backends.base.backends import vector_searcher_implementations
from superduperdb.backends.base.compute import ComputeBackend
from superduperdb.backends.base.data_backend import BaseDataBackend
from superduperdb.backends.base.metadata import MetaDataStore
from superduperdb.backends.base.query import Query
from superduperdb.backends.local.compute import LocalComputeBackend
from superduperdb.base import exceptions, variables
from superduperdb.base.config import Config
from superduperdb.base.cursor import SuperDuperCursor
from superduperdb.base.document import Document
from superduperdb.base.superduper import superduper
from superduperdb.cdc.cdc import DatabaseChangeDataCapture
from superduperdb.components.component import Component
from superduperdb.components.datatype import DataType, _BaseEncodable, serializers
from superduperdb.components.schema import Schema
from superduperdb.jobs.job import ComponentJob, FunctionJob, Job
from superduperdb.jobs.task_workflow import TaskWorkflow
from superduperdb.misc.annotations import deprecated
from superduperdb.misc.colors import Colors
from superduperdb.misc.data import ibatch
from superduperdb.misc.download import download_content, download_from_one
from superduperdb.vector_search.base import BaseVectorSearcher, VectorItem
from superduperdb.vector_search.interface import FastVectorSearcher
from superduperdb.vector_search.update_tasks import copy_vectors, delete_vectors

DBResult = t.Any
TaskGraph = t.Any

DeleteResult = DBResult
InsertResult = t.Tuple[DBResult, t.Optional[TaskGraph]]
SelectResult = SuperDuperCursor
UpdateResult = t.Any
PredictResult = t.Union[Document, t.Sequence[Document]]
ExecuteResult = t.Union[SelectResult, DeleteResult, UpdateResult, InsertResult]


class Datalayer:
    """
    Base database connector for SuperDuperDB.

    :param databackend: Object containing connection to Datastore.
    :param metadata: Object containing connection to Metadatastore.
    :param artifact_store: Object containing connection to Artifactstore.
    :param compute: Object containing connection to ComputeBackend.
    """

    type_id_to_cache_mapping = {
        'model': 'models',
        'metric': 'metrics',
        'datatype': 'datatypes',
        'vector_index': 'vector_indices',
        'schema': 'schemas',
        'listener': 'listeners',
    }
    cache_to_type_id_mapping = {v: k for k, v in type_id_to_cache_mapping.items()}

    def __init__(
        self,
        databackend: BaseDataBackend,
        metadata: MetaDataStore,
        artifact_store: ArtifactStore,
        compute: ComputeBackend = LocalComputeBackend(),
    ):
        """
        Initialize Data Layer.

        :param databackend: Object containing connection to Datastore.
        :param metadata: Object containing connection to Metadatastore.
        :param artifact_store: Object containing connection to Artifactstore.
        :param compute: Object containing connection to ComputeBackend.
        """
        logging.info("Building Data Layer")

        self.metrics = LoadDict(self, field='metric')
        self.models = LoadDict(self, field='model')
        self.datatypes = LoadDict(self, field='datatype')
        self.listeners = LoadDict(self, field='listener')
        self.vector_indices = LoadDict(self, field='vector_index')
        self.schemas = LoadDict(self, field='schema')
        self.tables = LoadDict(self, field='table')
        self.datatypes.update(serializers)

        self.fast_vector_searchers = LoadDict(
            self, callable=self.initialize_vector_searcher
        )
        self.metadata = metadata
        self.artifact_store = artifact_store
        self.artifact_store.serializers = self.datatypes

        self.databackend = databackend
        self.databackend.datalayer = self

        self.cdc = DatabaseChangeDataCapture(self)

        self.compute = compute
        self._server_mode = False
        self._cfg = s.CFG

    def __getitem__(self, item):
        return self.databackend.get_query_builder(item)

    @property
    def server_mode(self):
        """Property for server mode."""
        return self._server_mode

    @server_mode.setter
    def server_mode(self, is_server: bool):
        """
        Set server mode property.

        :param is_server: New boolean property.
        """
        assert isinstance(is_server, bool)
        self._server_mode = is_server

    def initialize_vector_searcher(
        self, identifier, searcher_type: t.Optional[str] = None
    ) -> t.Optional[BaseVectorSearcher]:
        """
        Initialize vector searcher.

        :param identifier: Identifying string to component.
        :param searcher_type: Searcher type (in_memory|native).
        """
        searcher_type = searcher_type or s.CFG.cluster.vector_search.type

        vi = self.vector_indices.force_load(identifier)
        from superduperdb import VectorIndex

        assert isinstance(vi, VectorIndex)

        clt = vi.indexing_listener.select.table_or_collection

        vector_search_cls = vector_searcher_implementations[searcher_type]
        vector_comparison = vector_search_cls.from_component(vi)

        assert isinstance(clt.identifier, str), 'clt.identifier must be a string'

        self.backfill_vector_search(vi, vector_comparison)

        return FastVectorSearcher(self, vector_comparison, vi.identifier)

    def backfill_vector_search(self, vi, searcher):
        """
        Backfill vector search from model outputs of a given vector index.

        :param vi: Identifier of vector index.
        :param searcher: FastVectorSearch instance to load model outputs as vectors.
        """
        if s.CFG.cluster.vector_search.type == 'native':
            return

        if s.CFG.cluster.vector_search.uri and not self.server_mode:
            return

        logging.info(f"Loading vectors of vector-index: '{vi.identifier}'")

        if vi.indexing_listener.select is None:
            raise ValueError('.select must be set')

        if vi.indexing_listener.select.db is None:
            vi.indexing_listener.select.db = self

        query = vi.indexing_listener.select.outputs(vi.indexing_listener.predict_id)

        logging.info(str(query))

        id_field = query.table_or_collection.primary_id

        progress = tqdm.tqdm(desc='Loading vectors into vector-table...')
        for record_batch in ibatch(
            self.execute(query),
            s.CFG.cluster.vector_search.backfill_batch_size,
        ):
            items = []
            for record in record_batch:
                id = record[id_field]
                assert not isinstance(vi.indexing_listener.model, str)
                h = record[f'_outputs.{vi.indexing_listener.predict_id}']
                if isinstance(h, _BaseEncodable):
                    h = h.unpack()
                items.append(VectorItem.create(id=str(id), vector=h))
            searcher.add(items)
            progress.update(len(items))

        searcher.post_create()

    # TODO - needed?
    def set_compute(self, new: ComputeBackend):
        """
        Set a new compute engine at runtime.

        Use it only if you know what you are doing.
        The standard procedure is to set the compute engine during initialization.

        :param new: New compute backend.
        """
        logging.warn(
            f"Changing compute engine from '{self.compute.name}' to '{new.name}'"
        )

        self.compute.disconnect()
        logging.success(
            f"Successfully disconnected from compute engine: '{self.compute.name}'"
        )

        logging.info(f"Connecting to compute engine: {new.name}")
        self.compute = new

    def get_compute(self):
        """Get compute."""
        return self.compute

    def drop(self, force: bool = False):
        """
        Drop all data, artifacts, and metadata.

        :param force: Force drop.
        """
        if not force and not click.confirm(
            f"{Colors.RED}[!!!WARNING USE WITH CAUTION AS YOU WILL"
            f"LOSE ALL DATA!!!]{Colors.RESET} "
            "Are you sure you want to drop the database? ",
            default=False,
        ):
            logging.warn("Aborting...")

        self.databackend.drop(force=True)
        self.metadata.drop(force=True)
        self.artifact_store.drop(force=True)

    def show(
        self,
        type_id: t.Optional[str] = None,
        identifier: t.Optional[str] = None,
        version: t.Optional[int] = None,
        include_presets: bool = False,
    ):
        """
        Show available functionality which has been added using ``self.add``.

        If the version is specified, then print full metadata.

        :param type_id: Type_id of component to show ['datatype', 'model', 'listener',
                       'learning_task', 'training_configuration', 'metric',
                       'vector_index', 'job'].
        :param identifier: Identifying string to component.
        :param version: (Optional) Numerical version - specify for full metadata.
        """
        if identifier is None and version is not None:
            raise ValueError(f"Must specify {identifier} to go with {version}")

        if type_id is None:
            nt = namedtuple('nt', ('type_id', 'identifier'))
            out = self.metadata.show_components()
            if not include_presets:
                return out
            subcaches = [
                self.models,
                self.datatypes,
                self.tables,
                self.schemas,
                self.vector_indices,
                self.listeners,
            ]
            cached = sum(
                [
                    [nt(comp.type_id, comp.identifier) for comp in subcache.values()]
                    for subcache in subcaches
                ],
                [],
            )
            out = sorted(list(set([nt(**x) for x in out] + cached)))
            return [x._asdict() for x in out]

        if identifier is None:
            mapping = self.type_id_to_cache_mapping
            out = self.metadata.show_components(type_id=type_id)
            if not include_presets:
                return sorted(out)
            try:
                t = mapping[type_id]
                return sorted(list(set(out + list(getattr(self, t).keys()))))
            except KeyError:
                return sorted(out)

        if version is None:
            return sorted(
                self.metadata.show_component_versions(
                    type_id=type_id, identifier=identifier
                )
            )

        if version == -1:
            return self._get_object_info(type_id=type_id, identifier=identifier)

        return self._get_object_info(
            type_id=type_id, identifier=identifier, version=version
        )

    def execute(self, query: Query, *args, **kwargs) -> ExecuteResult:
        """Execute a query on the database.

        :param query: The SQL query to execute, such as select, insert,
                      delete, or update.
        :param args: Positional arguments to pass to the execute call.
        :param kwargs: Keyword arguments to pass to the execute call.
        """
        if query.type == 'delete':
            return self._delete(query, *args, **kwargs)
        if query.type == 'insert':
            return self._insert(query, *args, **kwargs)
        if query.type == 'select':
            return self._select(query, *args, **kwargs)
        if query.type == 'write':
            return self._write(query, *args, **kwargs)
        if query.type == 'update':
            return self._update(query, *args, **kwargs)
        if query.type == 'predict':
            return self._predict(query, *args, **kwargs)

        raise TypeError(
            f'Wrong type of {query}; '
            f'Expected object of type "delete", "insert", "select", "update"'
            f'Got {type(query)};'
        )

    def _predict(self, prediction: t.Any) -> PredictResult:
        return prediction.do_execute(self)

    def _delete(self, delete: Query, refresh: bool = True) -> DeleteResult:
        """
        Delete data from the database.

        :param delete: The delete query object specifying the data to be deleted.
        """
        result = delete.do_execute(self)
        if refresh and not self.cdc.running:
            return result, self.refresh_after_delete(delete, ids=result)
        return result, None

    def _insert(
        self,
        insert: Query,
        refresh: bool = True,
        datatypes: t.Sequence[DataType] = (),
        auto_schema: bool = True,
    ) -> InsertResult:
        """
        Insert data into the database.

        :param insert: The insert query object specifying the data to be inserted.
        :param refresh: Boolean indicating whether to refresh the task group on insert.
        :param datatypes: List of datatypes in the insert documents.
        """
        for e in datatypes:
            self.add(e)

        for r in insert.documents:
            r['_fold'] = 'train'
            if random.random() < s.CFG.fold_probability:
                r['_fold'] = 'valid'

        if auto_schema and self.cfg.auto_schema:
            self.databackend.auto_create_table_schema(
                db=self, table_name=insert.identifier, documents=insert.documents
            )

        inserted_ids = insert.do_execute(self)

        cdc_status = self.cdc.running or s.CFG.cluster.cdc.uri is not None

        if refresh:
            if cdc_status:
                logging.warn('CDC service is active, skipping model/listener refresh')
            else:
                return inserted_ids, self.refresh_after_update_or_insert(
                    insert, ids=inserted_ids, verbose=False
                )

        return inserted_ids, None

    def _select(self, select: Query, reference: bool = True) -> SelectResult:
        """
        Select data from the database.

        :param select: The select query object specifying the data to be retrieved.
        """
        return select.do_execute(db=self)

    def refresh_after_delete(
        self,
        query: Query,
        ids: t.Sequence[str],
        verbose: bool = False,
    ):
        """
        Trigger cleanup jobs after data deletion.

        :param query: The select or update query object that reduces
                      the scope of computations.
        :param ids: IDs that further reduce the scope of computations.
        :param verbose: Set to ``True`` to enable more detailed output.
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
        query: Query,
        ids: t.Sequence[str],
        verbose: bool = False,
        overwrite: bool = False,
    ):
        """
        Trigger computation jobs after data insertion.

        :param query: The select or update query object that reduces
                      the scope of computations.
        :param ids: IDs that further reduce the scope of computations.
        :param verbose: Set to ``True`` to enable more detailed output.
        :param overwrite: If True, cascade the value to the 'predict_in_db' job.
        """
        task_workflow: TaskWorkflow = self._build_task_workflow(
            query.select_table,  # TODO can be replaced by select_using_ids
            ids=ids,
            verbose=verbose,
            overwrite=overwrite,
        )
        task_workflow.run_jobs()
        return task_workflow

    def _write(self, write: Query, refresh: bool = True) -> UpdateResult:
        """
        Bulk write data to the database.

        :param write: The update query object specifying the data to be written.
        :param refresh: Boolean indicating whether to refresh the task group on write.
        """
        write_result, updated_ids, deleted_ids = write.do_execute(self)

        cdc_status = self.cdc.running or s.CFG.cluster.cdc.uri is not None
        if refresh:
            if cdc_status:
                logging.warn('CDC service is active, skipping model/listener refresh')
            else:
                jobs = []
                for u, d in zip(write_result['update'], write_result['delete']):
                    q = u['query']
                    ids = u['ids']
                    # Overwrite should be true for update operation since updates
                    # could be done on collections with already existing outputs
                    # We need overwrite ouptuts on those select and recompute predict
                    job_update = self.refresh_after_update_or_insert(
                        query=q, ids=ids, verbose=False, overwrite=True
                    )
                    jobs.append(job_update)

                    q = d['query']
                    ids = d['ids']
                    job_update = self.refresh_after_delete(
                        query=q, ids=ids, verbose=False
                    )
                    jobs.append(job_update)

                return updated_ids, deleted_ids, jobs
        return updated_ids, deleted_ids, None

    def _update(self, update: Query, refresh: bool = True) -> UpdateResult:
        """
        Update data in the database.

        :param update: The update query object specifying the data to be updated.
        :param refresh: Boolean indicating whether to refresh the task group
                        after update.
        :return: Tuple containing the updated IDs and the refresh result if
                 performed.
        """
        updated_ids = update.do_execute(self)

        cdc_status = self.cdc.running or s.CFG.cluster.cdc.uri is not None
        if refresh:
            if cdc_status:
                logging.warn('CDC service is active, skipping model/listener refresh')
            else:
                # Overwrite should be true since updates could be done on collections
                # with already existing outputs.
                # We need overwrite outputs on those select and recompute predict
                return updated_ids, self.refresh_after_update_or_insert(
                    query=update, ids=updated_ids, verbose=False, overwrite=True
                )
        return updated_ids, None

    @deprecated
    def add(self, object: t.Any, dependencies: t.Sequence[Job] = ()):
        """
        Note: The use of `add` is deprecated, use `apply` instead.

        :param object: Object to be stored.
        :param dependencies: List of jobs which should execute before component
                             initialization begins.
        """
        return self.apply(object, dependencies=dependencies)

    def apply(
        self,
        object: t.Union[Component, t.Sequence[t.Any], t.Any],
        dependencies: t.Sequence[Job] = (),
    ):
        """
        Add functionality in the form of components.

        Components are stored in the configured artifact store
        and linked to the primary database through metadata.

        :param object: Object to be stored.
        :param dependencies: List of jobs which should execute before component
                             initialization begins.
        :return: Tuple containing the added object(s) and the original object(s).
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
        Remove a component (version optional).

        :param type_id: Type ID of the component to remove ('datatype',
                        'model', 'listener', 'training_configuration',
                        'vector_index').
        :param identifier: Identifier of the component (refer to
                            `container.base.Component`).
        :param version: [Optional] Numerical version to remove.
        :param force: Force skip confirmation (use with caution).
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
                uuid = self.metadata._get_component_uuid(type_id, identifier, v)
                component_versions_in_use.append(
                    f"{uuid} -> "
                    f"{self.metadata.get_component_version_parents(uuid)}",
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
        type_id: t.Optional[str] = None,
        identifier: t.Optional[str] = None,
        version: t.Optional[int] = None,
        allow_hidden: bool = False,
        uuid: t.Optional[str] = None,
        include_presets: bool = False,
    ) -> t.Union[Component, t.Dict[str, t.Any]]:
        """
        Load a component using uniquely identifying information.

        If `uuid` is provided, `type_id` and `identifier` are ignored.
        If `uuid` is not provided, `type_id` and `identifier` must be provided.

        :param type_id: Type ID of the component to load
                         ('datatype', 'model', 'listener', ...).
        :param identifier: Identifier of the component
                           (see `container.base.Component`).
        :param version: [Optional] Numerical version.
        :param allow_hidden: Toggle to ``True`` to allow loading
                             of deprecated components.
        :param uuid: [Optional] UUID of the component to load.
        :param include_presets: Include items cached in `db.models` etc.
        """
        if (
            include_presets
            and isinstance(type_id, str)
            and isinstance(identifier, str)
            and version is None
            and type_id in self.type_id_to_cache_mapping
        ):
            cache_name = self.type_id_to_cache_mapping[type_id]
            cache = getattr(self, cache_name)
            try:
                return cache[identifier]
            except KeyError:
                pass

        if type_id == 'encoder':
            logging.warn(
                '"encoder" has moved to "datatype" this functionality will not work'
                ' after version 0.2.0'
            )
            type_id = 'datatype'
        if uuid is None:
            if type_id is None or identifier is None:
                raise ValueError(
                    'Must specify `type_id` and `identifier` to load a component '
                    'when `uuid` is not provided.'
                )

            info = self.metadata.get_component(
                type_id=type_id,
                identifier=identifier,
                version=version,
                allow_hidden=allow_hidden,
            )
        else:
            # TODO: Do not call private method
            info = self.metadata._get_component_by_uuid(
                uuid=uuid,
                allow_hidden=allow_hidden,
            )
            assert info is not None
            type_id = info['type_id']
        m = Document.decode(info, db=self)
        m.db = self
        m.on_load(self)

        assert type_id is not None
        if cm := self.type_id_to_cache_mapping.get(type_id):
            try:
                getattr(self, cm)[m.identifier] = m
            except KeyError:
                raise exceptions.ComponentException('%s not found in %s cache'.format())
        return m

    def _build_delete_task_workflow(
        self,
        query: Query,
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
                s.CFG.cluster.vector_search.type == 'in_memory'
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
        overwrite: bool = False,
    ) -> TaskWorkflow:
        """A helper function to build a task workflow for a query with dependencies."""
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
                    query=query.encode(),
                ),
            ),
        )

        listeners = []
        for identifier in self.show('listener'):
            listener = self.listeners[identifier]
            if listener.select is None:
                continue

            if (
                listener.select.table_or_collection.identifier
                != query.table_or_collection.identifier
            ):
                continue

            listeners.append(listener)

        if not listeners:
            return G

        for listener in listeners:
            G.add_node(
                f'{listener.model.identifier}.predict_in_db({listener.uuid})',
                job=ComponentJob(
                    component_identifier=listener.model.identifier,
                    kwargs={
                        'X': listener.key,
                        'ids': ids,
                        'select': query.encode(),
                        'overwrite': overwrite,
                        'predict_id': listener.uuid,
                        **listener.predict_kwargs,
                    },
                    method_name='predict_in_db',
                    type_id='model',
                ),
            )

        for listener in listeners:
            G.add_edge(
                f'{download_content.__name__}()',
                f'{listener.model.identifier}.predict_in_db({listener.uuid})',
            )
            if not self.cdc.running:
                deps = listener.dependencies
                for dep in deps:
                    upstream = self.load(uuid=dep)
                    assert isinstance(upstream, Component)
                    dep_node = (
                        f'{upstream.model.identifier}.predict_in_db({upstream.uuid})'
                    )
                    if dep_node in G.nodes:
                        G.add_edge(
                            dep_node,
                            f'{listener.model.identifier}.predict_in_db({listener.uuid})',
                        )

        if s.CFG.cluster.vector_search.type == 'native':
            return G

        for identifier in self.show('vector_index'):
            # if a vector-searcher is not loaded, then skip
            # since s.CFG.cluster.vector_search_type == 'in_memory' implies the
            # program is standalone

            if (
                s.CFG.cluster.vector_search.uri is None
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
                        'query': vi.indexing_listener.select.encode(),
                    },
                ),
            )
            model = vi.indexing_listener.model.identifier
            uuid = vi.indexing_listener.uuid
            G.add_edge(
                f'{model}.predict_in_db({uuid})',
                f'{identifier}.{copy_vectors.__name__}',
            )

        return G

    def _add_child_components(self, components, parent):
        # TODO this is a bit of a mess
        # it handles the situation in `Stack` when
        # the components should be added in a certain order
        G = networkx.DiGraph()
        lookup = {(c.type_id, c.identifier): c for c in components}
        for k in lookup:
            G.add_node(k)
            for d in lookup[k].dependencies:
                if d[:2] in lookup:
                    G.add_edge(d, lookup[k].id_tuple)

        nodes = networkx.topological_sort(G)
        jobs = {}
        for n in nodes:
            component = lookup[n]
            dependencies = sum(
                [jobs.get(d[:2], []) for d in component.dependencies], []
            )
            tmp = self._add(component, parent=parent.uuid, dependencies=dependencies)
            jobs[n] = tmp

        return sum(list(jobs.values()), [])

    def _update_component(
        self, object, dependencies: t.Sequence[Job] = (), parent: t.Optional[str] = None
    ):
        # TODO add update logic here to check changed attributes
        s.logging.debug(f'{object.id} already exists - doing nothing')
        return []

    def _add(
        self,
        object: Component,
        dependencies: t.Sequence[Job] = (),
        parent: t.Optional[str] = None,
    ):
        jobs = list(dependencies)
        object.db = self
        object.pre_create(self)
        assert hasattr(object, 'identifier')
        assert hasattr(object, 'version')

        if not isinstance(object.identifier, variables.Variable):
            existing_versions = self.show(object.type_id, object.identifier)
        else:
            existing_versions = []
        already_exists = (
            isinstance(object.version, int) and object.version in existing_versions
        )

        if already_exists:
            return self._update_component(
                object, dependencies=dependencies, parent=parent
            )

        if object.version is None:
            if existing_versions:
                object.version = max(existing_versions) + 1
            else:
                object.version = 0

        serialized = object.dict().encode(leaves_to_keep=(Component,))

        children = [
            v for v in serialized['_leaves'].values() if isinstance(v, Component)
        ]

        jobs.extend(self._add_child_components(children, parent=object))

        for k, v in serialized['_leaves'].items():
            if isinstance(v, Component):
                serialized['_leaves'][
                    k
                ] = f'?db.load({v.type_id}, {v.identifier}, {v.version})'

        serialized = self.artifact_store.save_artifact(serialized)

        self.metadata.create_component(serialized)

        if parent is not None:
            self.metadata.create_parent_child(parent, object.uuid)

        object.post_create(self)
        self._add_component_to_cache(object)
        these_jobs = object.schedule_jobs(self, dependencies=dependencies)
        jobs.extend(these_jobs)
        return jobs

    def _remove_component_version(
        self,
        type_id: str,
        identifier: str,
        version: int,
        force: bool = False,
    ):
        r = self.metadata.get_component(type_id, identifier, version=version)
        if self.metadata.component_version_has_parents(type_id, identifier, version):
            parents = self.metadata.get_component_version_parents(r['uuid'])
            raise Exception(f'{r["uuid"]} is involved in other components: {parents}')

        if force or click.confirm(
            f'You are about to delete {type_id}/{identifier}{version}, are you sure?',
            default=False,
        ):
            component = self.load(
                type_id, identifier, version=version, allow_hidden=force
            )
            info = self.metadata.get_component(
                type_id, identifier, version=version, allow_hidden=force
            )
            if hasattr(component, 'cleanup'):
                # TODO - is there an abstract method thingy for this?
                component.cleanup(self)

            if type_id in self.type_id_to_cache_mapping:
                try:
                    del getattr(self, self.type_id_to_cache_mapping[type_id])[
                        identifier
                    ]
                except KeyError:
                    pass

            self.artifact_store.delete_artifact(info)
            self.metadata.delete_component_version(type_id, identifier, version=version)

    def _get_content_for_filter(self, filter) -> Document:
        if isinstance(filter, dict):
            filter = Document(filter)
        if '_id' not in filter:
            filter['_id'] = 0
        download_from_one(filter)
        if not filter['_id']:
            del filter['_id']
        return filter

    # # TODO should create file handler separately
    # def _get_file_content(self, r):
    #     for k in r:
    #         if isinstance(r[k], dict):
    #             r[k] = self._get_file_content(r[k])
    #     return r

    # TODO should be confined to metadata implementation
    def _get_object_info(self, identifier, type_id, version=None):
        return self.metadata.get_component(type_id, identifier, version=version)

    def replace(
        self,
        object: t.Any,
        upsert: bool = False,
    ):
        """
        Replace a model in the artifact store with an updated object.

        (Use with caution!)

        :param object: The object to replace.
        :param upsert: Toggle to ``True`` to enable replacement even if
                       the object doesn't exist yet.
        """
        old_uuid = None
        try:
            info = self.metadata.get_component(
                object.type_id, object.identifier, version=object.version
            )
            old_uuid = info['uuid']
        except FileNotFoundError as e:
            if upsert:
                return self.apply(
                    object,
                )
            raise e

        # If object has no version, update the last version
        object.version = info['version']

        serialized = object.dict().encode(leaves_to_keep=(Component,))

        children = [
            v for v in serialized['_leaves'].values() if isinstance(v, Component)
        ]

        for child in children:
            self.replace(child, upsert=True)
            if old_uuid:
                self.metadata.delete_parent_child(old_uuid, child.uuid)

        for k, v in serialized['_leaves'].items():
            if isinstance(v, Component):
                serialized['_leaves'][
                    k
                ] = f'?db.load({v.type_id}, {v.identifier}, {v.version})'

        self.artifact_store.delete_artifact(info)

        serialized = self.artifact_store.save_artifact(serialized)

        self.metadata.replace_object(
            serialized,
            identifier=object.identifier,
            type_id=object.type_id,
            version=object.version,
        )

    def select_nearest(
        self,
        like: t.Union[t.Dict, Document],
        vector_index: str,
        ids: t.Optional[t.Sequence[str]] = None,
        outputs: t.Optional[Document] = None,
        n: int = 100,
    ) -> t.Tuple[t.List[str], t.List[float]]:
        """
        Performs a vector search query on the given vector index.

        :param like: Vector search document to search.
        :param vector_index: Vector index to search.
        :param ids: (Optional) IDs to search within.
        :param outputs: (Optional) Seed outputs dictionary.
        :param n: Get top k results from vector search.
        """
        # TODO - make this un-ambiguous
        if not isinstance(like, Document):
            assert isinstance(like, dict)
            like = Document(like)
        like = self._get_content_for_filter(like)
        vi = self.vector_indices[vector_index]
        if outputs is None:
            outs: t.Dict = {}
        else:
            outs = outputs.encode()
            if not isinstance(outs, dict):
                raise TypeError(f'Expected dict, got {type(outputs)}')
        logging.info(str(outs))
        return vi.get_nearest(like, db=self, ids=ids, n=n, outputs=outs)

    def close(self):
        """Gracefully shutdown the Datalayer."""
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

    def _add_component_to_cache(self, component: Component):
        """
        Add component to cache when it is added to the db.

        Avoiding the need to load it from the db again.
        """
        type_id = component.type_id
        if cm := self.type_id_to_cache_mapping.get(type_id):
            getattr(self, cm)[component.identifier] = component
        component.on_load(self)

    def infer_schema(
        self, data: t.Mapping[str, t.Any], identifier: t.Optional[str] = None
    ) -> Schema:
        """Infer a schema from a given data object.

        :param data: The data object
        :param identifier: The identifier for the schema, if None, it will be generated
        :return: The inferred schema
        """
        return self.databackend.infer_schema(data, identifier)

    @property
    def cfg(self) -> Config:
        """Get the configuration object for the datalayer."""
        return self._cfg or s.CFG

    @cfg.setter
    def cfg(self, cfg: Config):
        """Set the configuration object for the datalayer."""
        assert isinstance(cfg, Config)
        self._cfg = cfg


@dc.dataclass
class LoadDict(dict):
    """
    Helper class to load component identifiers with on-demand loading from the database.

    :param database: Instance of Datalayer.
    :param field: (optional) Component type identifier.
    :param callable: (optional) Callable function on key.
    """

    database: Datalayer
    field: t.Optional[str] = None
    callable: t.Optional[t.Callable] = None

    def __missing__(self, key: str):
        if self.field is not None:
            value = self[key] = self.database.load(
                self.field, key, include_presets=False
            )
        else:
            msg = f'callable is ``None`` for {key}'
            assert self.callable is not None, msg
            value = self[key] = self.callable(key)
        return value

    def force_load(self, key: str):
        """
        Force load the component from database.

        :param key: Force load key
        """
        return self.__missing__(key)
