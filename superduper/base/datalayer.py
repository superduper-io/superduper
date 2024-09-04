import dataclasses as dc
import random
import typing as t
import warnings
from collections import defaultdict, namedtuple

import click
import networkx

import superduper as s
from superduper import CFG, logging
from superduper.backends.base.artifacts import ArtifactStore
from superduper.backends.base.backends import vector_searcher_implementations
from superduper.backends.base.compute import ComputeBackend
from superduper.backends.base.data_backend import BaseDataBackend
from superduper.backends.base.metadata import MetaDataStore
from superduper.backends.base.query import Query
from superduper.backends.local.compute import LocalComputeBackend
from superduper.base import exceptions
from superduper.base.config import Config
from superduper.base.constant import KEY_BUILDS
from superduper.base.cursor import SuperDuperCursor
from superduper.base.document import Document
from superduper.base.event import Event
from superduper.components.component import Component
from superduper.components.datatype import DataType
from superduper.components.schema import Schema
from superduper.components.table import Table
from superduper.jobs.job import Job
from superduper.misc.annotations import deprecated
from superduper.misc.colors import Colors
from superduper.misc.download import download_from_one
from superduper.misc.retry import db_retry
from superduper.misc.special_dicts import recursive_update
from superduper.vector_search.base import BaseVectorSearcher
from superduper.vector_search.interface import FastVectorSearcher

DBResult = t.Any
TaskGraph = t.Any

DeleteResult = DBResult
InsertResult = t.Tuple[DBResult, t.Optional[TaskGraph]]
SelectResult = SuperDuperCursor
UpdateResult = t.Any
PredictResult = t.Union[Document, t.Sequence[Document]]
ExecuteResult = t.Union[SelectResult, DeleteResult, UpdateResult, InsertResult]


@dc.dataclass
class DBEvent:
    """Event to represent database events."""

    insert = 'insert'
    delete = 'delete'
    update = 'update'
    upsert = 'upsert'

    @staticmethod
    def chunk_by_event(lst: t.List[Event]) -> t.Dict[str, t.List[Event]]:
        """Helper method to chunk events on type."""
        chunks = defaultdict(list)
        for item in lst:
            item_type = item.event_type
            chunks[item_type].append(item)
        return chunks


class Datalayer:
    """
    Base database connector for superduper.io.

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

        self.fast_vector_searchers = LoadDict(
            self, callable=self.initialize_vector_searcher
        )
        self.metadata = metadata
        self.artifact_store = artifact_store
        self.artifact_store.serializers = self.datatypes

        self.databackend = databackend
        self.databackend.datalayer = self

        self._cdc = None

        self.compute = compute
        self.compute.queue.db = self
        self._cfg = s.CFG

    def __getitem__(self, item):
        return self.databackend.get_query_builder(item)

    @property
    def cdc(self):
        """CDC property."""
        return self._cdc

    @cdc.setter
    def cdc(self, cdc):
        """CDC property setter."""
        self._cdc = cdc

    def initialize_vector_searcher(
        self, vi, searcher_type: t.Optional[str] = None
    ) -> t.Optional[BaseVectorSearcher]:
        """
        Initialize vector searcher.

        :param vi: Identifying string to component.
        :param searcher_type: Searcher type (in_memory|native).
        """
        searcher_type = searcher_type or s.CFG.cluster.vector_search.type

        logging.info(f"Initializing vector searcher with type {searcher_type}")
        if isinstance(vi, str):
            vi = self.vector_indices.force_load(vi)
        from superduper import VectorIndex

        assert isinstance(vi, VectorIndex)

        assert vi.indexing_listener.select is not None
        clt = vi.indexing_listener.select.table_or_collection

        vector_search_cls = vector_searcher_implementations[searcher_type]
        logging.info(f"Using vector searcher: {vector_search_cls}")
        vector_comparison = vector_search_cls.from_component(vi)

        assert isinstance(clt.identifier, str), 'clt.identifier must be a string'
        return FastVectorSearcher(self, vector_comparison, vi.identifier)

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

    def disconnect(self):
        """Disconnect from the compute engine."""
        self.compute.disconnect()

    def get_compute(self):
        """Get compute."""
        return self.compute

    def drop(self, force: bool = False, data: bool = False):
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

        if self._cfg.cluster.vector_search.uri:
            for vi in self.show('vector_index'):
                FastVectorSearcher.drop_remote(vi)

        if data:
            self.databackend.drop(force=True)
            self.artifact_store.drop(force=True)
        else:
            self.databackend.drop_outputs()
        self.metadata.drop(force=True)

    def show(
        self,
        type_id: t.Optional[str] = None,
        identifier: t.Optional[str] = None,
        version: t.Optional[int] = None,
        uuid: t.Optional[str] = None,
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

        if uuid is not None:
            return self.metadata.get_component_by_uuid(uuid)

        if identifier is None and version is not None:
            raise ValueError(f"Must specify {identifier} to go with {version}")

        if type_id is None:
            nt = namedtuple('nt', ('type_id', 'identifier'))
            out = self.metadata.show_components()
            out = sorted(list(set([nt(**x) for x in out])))
            return [x._asdict() for x in out]

        if identifier is None:
            out = self.metadata.show_components(type_id=type_id)
            return sorted(out)

        if version is None:
            out = sorted(
                self.metadata.show_component_versions(
                    type_id=type_id, identifier=identifier
                )
            )
            return out

        if version == -1:
            return self.metadata.get_component(
                type_id=type_id, identifier=identifier, version=None
            )

        return self.metadata.get_component(
            type_id=type_id, identifier=identifier, version=version
        )

    @db_retry(connector='databackend')
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
        cdc_status = s.CFG.cluster.cdc.uri is not None
        if refresh and not cdc_status:
            return result, self.on_event(delete, ids=result, event_type=DBEvent.delete)
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
            r.setdefault(
                '_fold',
                'train' if random.random() >= s.CFG.fold_probability else 'valid',
            )

        if auto_schema and self.cfg.auto_schema:
            self._auto_create_table(insert.table, insert.documents)

        inserted_ids = insert.do_execute(self)

        cdc_status = s.CFG.cluster.cdc.uri is not None
        is_output_table = insert.table.startswith(CFG.output_prefix)

        logging.info(f'Inserted {len(inserted_ids)} documents into {insert.table}')
        logging.debug(f'Inserted IDs: {inserted_ids}')

        if refresh:
            if cdc_status and not is_output_table:
                logging.warn('CDC service is active, skipping model/listener refresh')
            else:
                return inserted_ids, self.on_event(
                    insert,
                    ids=inserted_ids,
                )

        return inserted_ids, None

    def _auto_create_table(self, table_name, documents):
        try:
            table = self.tables[table_name]
            return table
        except FileNotFoundError:
            logging.info(f"Table {table_name} does not exist, auto creating...")

        # Should we need to check all the documents?
        document = documents[0]
        schema = document.schema or self.infer_schema(document)
        table = Table(identifier=table_name, schema=schema)
        logging.info(f"Creating table {table_name} with schema {schema.fields_set}")
        self.apply(table)

    def _select(self, select: Query, reference: bool = True) -> SelectResult:
        """
        Select data from the database.

        :param select: The select query object specifying the data to be retrieved.
        """
        return select.do_execute(db=self)

    def on_event(self, query: Query, ids: t.List[str], event_type: str = 'insert'):
        """
        Trigger computation jobs after data insertion.

        :param query: The select or update query object that reduces
                      the scope of computations.
        :param ids: IDs that further reduce the scope of computations.
        """
        event_datas = query.get_events(ids)
        if not event_datas:
            return

        from superduper.base.event import Event

        events = []
        for event_data in event_datas:
            identifier = event_data['identifier']
            type_id = event_data['type_id']
            ids = event_data['ids']
            dest = {'identifier': identifier, 'type_id': type_id}
            events.extend(
                [Event(dest=dest, id=str(id), event_type=event_type) for id in ids]
            )

        logging.info(
            f'Created {len(events)} events for {event_type} on [{query.table}]'
        )
        logging.info(f'Broadcasting {len(events)} events')
        return self.compute.broadcast(events)

    def _write(self, write: Query, refresh: bool = True) -> UpdateResult:
        """
        Bulk write data to the database.

        :param write: The update query object specifying the data to be written.
        :param refresh: Boolean indicating whether to refresh the task group on write.
        """
        write_result, updated_ids, deleted_ids = write.do_execute(self)

        cdc_status = s.CFG.cluster.cdc.uri is not None
        if refresh:
            if cdc_status:
                logging.warn('CDC service is active, skipping model/listener refresh')
            else:
                jobs = []
                if updated_ids:
                    job = self.on_event(
                        query=write, ids=updated_ids, event_type=DBEvent.update
                    )
                    jobs.append(job)
                if deleted_ids:
                    job = self.on_event(
                        query=write, ids=deleted_ids, event_type=DBEvent.delete
                    )
                    jobs.append(job)

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

        cdc_status = s.CFG.cluster.cdc.uri is not None
        is_output_table = update.table.startswith(CFG.output_prefix)
        if refresh and updated_ids:
            if cdc_status and not is_output_table:
                logging.warn('CDC service is active, skipping model/listener refresh')
            else:
                # Overwrite should be true since updates could be done on collections
                # with already existing outputs.
                # We need overwrite outputs on those select and recompute predict
                return updated_ids, self.on_event(
                    query=update, ids=updated_ids, event_type=DBEvent.upsert
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
        if not isinstance(object, Component):
            raise ValueError('Only components can be applied')
        return self._apply(object=object, dependencies=dependencies), object

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
    ) -> Component:
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
        """
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
            info = self.metadata.get_component_by_uuid(
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

    def _add_child_components(self, components, parent, parent_deps):
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
                [jobs.get(d[:2], []) for d in component.dependencies], list(parent_deps)
            )
            tmp = self._apply(component, parent=parent.uuid, dependencies=dependencies)
            jobs[n] = tmp
        return sum(list(jobs.values()), [])

    def _update_component(
        self, object, dependencies: t.Sequence[Job] = (), parent: t.Optional[str] = None
    ):
        # TODO add update logic here to check changed attributes
        s.logging.debug(
            f'{object.type_id},{object.identifier} already exists - doing nothing'
        )
        return []

    def _apply(
        self,
        object: Component,
        dependencies: t.Sequence[Job] = (),
        parent: t.Optional[str] = None,
        artifacts: t.Optional[t.Dict[str, bytes]] = None,
    ):
        jobs = list(dependencies)
        object.db = self
        object.pre_create(self)
        assert hasattr(object, 'identifier')
        assert hasattr(object, 'version')

        existing_versions = self.show(object.type_id, object.identifier)

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

        for k, v in serialized[KEY_BUILDS].items():
            if isinstance(v, Component) and hasattr(v, 'inline') and v.inline:
                r = dict(v.dict())
                del r['identifier']
                serialized[KEY_BUILDS][k] = r

        children = [
            v for v in serialized[KEY_BUILDS].values() if isinstance(v, Component)
        ]
        child_jobs = self._add_child_components(
            children, parent=object, parent_deps=dependencies
        )
        jobs.extend(child_jobs)

        if children:
            serialized = self._change_component_reference_prefix(serialized)

        serialized = self._save_artifact(object.uuid, serialized)
        if artifacts:
            for file_id, bytes in artifacts.items():
                self.artifact_store.put_bytes(bytes, file_id)

        self.metadata.create_component(serialized)

        if parent is not None:
            self.metadata.create_parent_child(parent, object.uuid)

        deps = []
        for job in jobs:
            if isinstance(job, Job):
                deps.append(job.job_id)
            else:
                deps.append(job)
        dependencies = list(set([*deps, *dependencies]))  # type: ignore[list-item]

        object.post_create(self)
        these_jobs = object.schedule_jobs(self, dependencies=dependencies)
        jobs.extend(these_jobs)

        self._add_component_to_cache(object)
        return jobs

    def _change_component_reference_prefix(self, serialized):
        """Replace '?' to '&' in the serialized object."""
        references = {}
        for reference in list(serialized[KEY_BUILDS].keys()):
            if isinstance(serialized[KEY_BUILDS][reference], Component):
                comp = serialized[KEY_BUILDS][reference]
                serialized[KEY_BUILDS].pop(reference)
                references[reference] = (
                    comp.type_id + ':' + comp.identifier + ':' + comp.uuid
                )

        # Only replace component references
        if not references:
            return

        def replace_function(value):
            # Change value if it is a string and starts with '?'
            # and the value is in references
            # ?:xxx: -> &:xxx:
            if (
                isinstance(value, str)
                and value.startswith('?')
                and value[1:] in references
            ):
                return '&:component:' + references[value[1:]]
            return value

        serialized = recursive_update(serialized, replace_function)
        return serialized

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
            component.cleanup(self)

            if type_id in self.type_id_to_cache_mapping:
                try:
                    del getattr(self, self.type_id_to_cache_mapping[type_id])[
                        identifier
                    ]
                except KeyError:
                    pass

            self._delete_artifacts(r['uuid'], info)
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
        for k, v in serialized[KEY_BUILDS].items():
            if isinstance(v, Component) and hasattr(v, 'inline') and v.inline:
                r = dict(v.dict())
                del r['identifier']
                serialized[KEY_BUILDS][k] = r

        children = [
            v for v in serialized[KEY_BUILDS].values() if isinstance(v, Component)
        ]

        for child in children:
            self.replace(child, upsert=True)
            if old_uuid:
                self.metadata.delete_parent_child(old_uuid, child.uuid)

        if children:
            serialized = self._change_component_reference_prefix(serialized)

        self._delete_artifacts(object.uuid, info)

        serialized = self._save_artifact(object.uuid, serialized)

        self.metadata.replace_object(
            serialized,
            identifier=object.identifier,
            type_id=object.type_id,
            version=object.version,
        )

    def _save_artifact(self, uuid, info: t.Dict):
        """
        Save an artifact to the artifact store.

        :param artifact: The artifact to save.
        """
        artifact_ids, _ = self._find_artifacts(info)
        self.metadata.create_artifact_relation(uuid, artifact_ids)
        return self.artifact_store.save_artifact(info)

    def _delete_artifacts(self, uuid, info: t.Dict):
        artifact_ids, artifacts = self._find_artifacts(info)
        for artifact_id in artifact_ids:
            relation_uuids = self.metadata.get_artifact_relations(
                artifact_id=artifact_id
            )
            if len(relation_uuids) == 1 and relation_uuids[0] == uuid:
                self.artifact_store.delete_artifact([artifact_id])
                self.metadata.delete_artifact_relation(
                    uuid=uuid, artifact_ids=artifact_id
                )

    def _find_artifacts(self, info: t.Dict):
        from superduper.misc.special_dicts import recursive_find

        # find all blobs with `&:blob:` prefix,
        blobs = recursive_find(
            info, lambda v: isinstance(v, str) and v.startswith('&:blob:')
        )

        # find all files with `&:file:` prefix
        files = recursive_find(
            info, lambda v: isinstance(v, str) and v.startswith('&:file:')
        )
        artifact_ids: list[str] = []
        artifact_ids.extend(a.split(":")[-1] for a in blobs)
        artifact_ids.extend(a.split(":")[-1] for a in files)
        return artifact_ids, {'blobs': blobs, 'files': files}

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

    def _add_component_to_cache(self, component: Component):
        """
        Add component to cache when it is added to the db.

        Avoiding the need to load it from the db again.
        """
        type_id = component.type_id
        if cm := self.type_id_to_cache_mapping.get(type_id):
            # NOTE: We need to reload the object since in `schedule_jobs`
            # of the object, `db.replace` might be performed.
            # e.g model prediction object is replace with updated datatype.
            self.load(type_id, component.identifier)
            getattr(self, cm)[component.identifier] = component

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

    def __missing__(self, key: t.Union[str, Component]):
        if self.field is not None:
            key = key.identifier if isinstance(key, Component) else key
            value = self[key] = self.database.load(
                self.field,
                key,
            )
        else:
            msg = f'callable is ``None`` for {key}'
            assert self.callable is not None, msg
            value = self.callable(key)
            key = key.identifier if isinstance(key, Component) else key
            self[key] = value
        return value

    def force_load(self, key: str):
        """
        Force load the component from database.

        :param key: Force load key
        """
        return self.__missing__(key)
