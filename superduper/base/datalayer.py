import random
import typing as t
import warnings
from collections import namedtuple

import click
import networkx

import superduper as s
from superduper import CFG, logging
from superduper.backends.base.artifacts import ArtifactStore
from superduper.backends.base.cluster import Cluster
from superduper.backends.base.compute import ComputeBackend
from superduper.backends.base.data_backend import BaseDataBackend
from superduper.backends.base.metadata import MetaDataStore
from superduper.backends.base.query import Query
from superduper.base import exceptions
from superduper.base.config import Config
from superduper.base.constant import KEY_BUILDS
from superduper.base.cursor import SuperDuperCursor
from superduper.base.document import Document
from superduper.base.event import Create, Signal
from superduper.components.component import Component, Status
from superduper.components.datatype import DataType
from superduper.components.schema import Schema
from superduper.components.table import Table
from superduper.misc.annotations import deprecated
from superduper.misc.colors import Colors
from superduper.misc.download import download_from_one
from superduper.misc.retry import db_retry
from superduper.misc.special_dicts import recursive_update

if t.TYPE_CHECKING:
    from superduper.base.event import Job


DBResult = t.Any
TaskGraph = t.Any

DeleteResult = DBResult
InsertResult = t.List[str]
SelectResult = SuperDuperCursor
UpdateResult = t.List[str]
PredictResult = t.Union[Document, t.Sequence[Document]]
ExecuteResult = t.Union[SelectResult, DeleteResult, UpdateResult, InsertResult]


class Datalayer:
    """
    Base database connector for superduper.io.

    :param databackend: Object containing connection to Datastore.
    :param metadata: Object containing connection to Metadatastore.
    :param artifact_store: Object containing connection to Artifactstore.
    :param cluster: Cluster object containing connections to infrastructure.
    """

    def __init__(
        self,
        databackend: BaseDataBackend,
        metadata: MetaDataStore,
        artifact_store: ArtifactStore,
        cluster: Cluster,
    ):
        """
        Initialize Data Layer.

        :param databackend: Object containing connection to Datastore.
        :param metadata: Object containing connection to Metadatastore.
        :param artifact_store: Object containing connection to Artifactstore.
        :param compute: Object containing connection to ComputeBackend.
        """
        logging.info("Building Data Layer")

        self.metadata = metadata

        self.artifact_store = artifact_store
        self.artifact_store.db = self

        self.databackend = databackend
        self.databackend.datalayer = self

        self.cluster = cluster
        self.cluster.db = self

        self._cfg = s.CFG
        self.startup_cache: t.Dict[str, t.Any] = {}

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

    # TODO needed?
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

        # drop the cache, vector-indexes, triggers, queues
        self.cluster.drop(force=True)

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
        # TODO - do we need this refresh?
        # If the handle-event works well, then we should not need this

        if not refresh:
            return

        call_cdc = (
            delete.query.table in self.metadata.show_cdc_tables()
            and delete.query.table.startswith(CFG.output_prefix)
        )
        if call_cdc:
            self.cluster.cdc.handle_event(
                event_type='delete', table=delete.table, ids=result
            )
        return result

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
        :param auto_schema: Toggle to False to switch off automatic schema creation.
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

            timeout = 5

            import time

            start = time.time()

            exists = False
            while time.time() - start < timeout:
                try:
                    assert insert.table in self.show(
                        'table'
                    ), f'{insert.table} not found, retrying...'
                    exists = True
                except AssertionError as e:
                    logging.warn(str(e))
                    time.sleep(0.25)
                    continue
                break

            if not exists:
                raise TimeoutError(
                    f'{insert.table} not found after {timeout} seconds'
                    ' table auto creation likely has failed or is stalling...'
                )

        inserted_ids = insert.do_execute(self)

        logging.info(f'Inserted {len(inserted_ids)} documents into {insert.table}')
        logging.debug(f'Inserted IDs: {inserted_ids}')

        if not refresh:
            return []

        if (
            insert.table in self.metadata.show_cdc_tables()
            and not insert.table.startswith(CFG.output_prefix)
        ):
            self.cluster.cdc.handle_event(
                event_type='insert', table=insert.table, ids=inserted_ids
            )

        return inserted_ids

    def _auto_create_table(self, table_name, documents):
        try:
            table = self.load('table', table_name)
            return table
        except FileNotFoundError:
            logging.info(f"Table {table_name} does not exist, auto creating...")

        # Should we need to check all the documents?
        document = documents[0]
        schema = document.schema or self.infer_schema(document)
        table = Table(identifier=table_name, schema=schema)
        logging.info(f"Creating table {table_name} with schema {schema.fields_set}")
        self.apply(table, force=True)

    def _select(self, select: Query, reference: bool = True) -> SelectResult:
        """
        Select data from the database.

        :param select: The select query object specifying the data to be retrieved.
        """
        return select.do_execute(db=self)

    def on_event(self, table: str, ids: t.List[str], event_type: 'str'):
        """
        Trigger computation jobs after data insertion.

        :param query: The select or update query object that reduces
                      the scope of computations.
        :param ids: IDs that further reduce the scope of computations.
        """
        from superduper.base.event import Change

        component_exists_to_consume = False
        for component in self.show("listener"):
            cdc_table = self.load('listener', component).cdc_table
            if cdc_table == table:
                component_exists_to_consume = True
                break
        if not component_exists_to_consume:
            logging.info(
                'Skipping cdc for inserted documents in {table}',
                'because no component to consume the table.',
            )
            return

        events = []
        for id in ids:
            event = Change(ids=[str(id)], queue=table, type=event_type)
            events.append(event)
        logging.info(f'Created {len(events)} events for {event_type} on [{table}]')
        logging.info(f'Publishing {len(events)} events')
        return self.cluster.queue.publish(events)  # type: ignore[arg-type]

    def _write(self, write: Query, refresh: bool = True) -> UpdateResult:
        """
        Bulk write data to the database.

        :param write: The update query object specifying the data to be written.
        :param refresh: Boolean indicating whether to refresh the task group on write.
        """
        _, updated_ids, deleted_ids = write.do_execute(self)

        if not refresh:
            return []

        call_cdc = (
            write.table in self.metadata.show_cdc_tables()
            and write.table.startswith(CFG.output_prefix)
        )
        if call_cdc:
            if updated_ids:
                self.cluster.cdc.handle_event(
                    event_type='update', table=write.table, ids=updated_ids
                )
            if deleted_ids:
                self.cluster.cdc.handle_event(
                    event_type='delete', table=write.table, ids=deleted_ids
                )

        return updated_ids + deleted_ids

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

        if not refresh:
            return updated_ids

        table = update.table

        if table in self.metadata.show_cdc_tables() and not table.startswith(
            CFG.output_prefix
        ):
            self.cluster.cdc.handle_event(
                event_type='update', table=update.table, ids=updated_ids
            )

        return updated_ids

    @deprecated
    def add(self, object: t.Any):
        """
        Note: The use of `add` is deprecated, use `apply` instead.

        :param object: Object to be stored.
        :param dependencies: List of jobs which should execute before component
                             initialization begins.
        """
        return self.apply(object)

    def apply(
        self,
        object: t.Union[Component, t.Sequence[t.Any], t.Any],
        force: bool | None = None,
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
        if force is None:
            force = self.cfg.force_apply

        if not isinstance(object, Component):
            raise ValueError('Only components can be applied')

        # This populates the component with data fetched
        # from `db` if necessary
        # We need pre as well as post-create, since the order
        # between parents and children are reversed in each
        # sometimes parents might need to grab things from children
        # and vice-versa
        object.pre_create(self)

        # context allows us to track the origin of the component creation
        create_events, job_events = self._apply(
            object=object,
            context=object.uuid,
            job_events=[],
        )
        # this flags that the context is not needed anymore
        if not create_events:
            return object

        # TODO for some reason the events get created multiple times
        # we need to fix that to prevent inefficiencies
        unique_create_ids = []
        unique_create_events = []
        for e in create_events:
            if e.component['uuid'] not in unique_create_ids:
                unique_create_ids.append(e.component['uuid'])
                unique_create_events.append(e)

        unique_job_ids = []
        unique_job_events = []
        for e in job_events:
            if e.job_id not in unique_job_ids:
                unique_job_ids.append(e.job_id)
                unique_job_events.append(e)

        logging.info('Here are the CREATION EVENTS:')
        steps = {
            c.component['uuid']: str(i) for i, c in enumerate(unique_create_events)
        }
        for i, c in enumerate(unique_create_events):
            if c.parent:
                logging.info(f'[{i}]: {c.huuid}: create ~ [{steps[c.parent]}]')
            else:
                logging.info(f'[{i}]: {c.huuid}: create')

        logging.info('JOBS EVENTS:')
        steps = {j.job_id: str(i) for i, j in enumerate(unique_job_events)}

        def uniquify(x):
            return sorted(list(set(x)))

        for i, j in enumerate(unique_job_events):
            if j.dependencies:
                logging.info(
                    f'[{i}]: {j.huuid}: {j.method} ~ '
                    f'[{",".join(uniquify([steps[d] for d in j.dependencies]))}]'
                )
            else:
                logging.info(f'[{i}]: {j.huuid}: {j.method}')

        events = [
            *unique_create_events,
            *unique_job_events,
            Signal(context=object.uuid, msg='done'),
        ]

        if not force:
            if not click.confirm(
                '\033[1mPlease approve this deployment plan.\033[0m',
                default=True,
            ):
                return object
        self.cluster.queue.publish(events=events)
        return object

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
        uuid: t.Optional[str] = None,
        huuid: t.Optional[str] = None,
        on_load: bool = True,
        allow_hidden: bool = False,
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
        if version is not None:
            assert type_id is not None
            assert identifier is not None
            info = self.metadata.get_component(
                type_id=type_id,
                identifier=identifier,
                version=version,
                allow_hidden=allow_hidden,
            )
            uuid = info['uuid']

        if huuid is not None:
            uuid = huuid.split(':')[-1]

        if uuid is not None:
            try:
                return self.cluster.cache[uuid]
            except KeyError:
                logging.info(f'Component {uuid} not found in cache, loading from db')
                info = self.metadata.get_component_by_uuid(
                    uuid=uuid, allow_hidden=allow_hidden
                )
                c = Document.decode(info, db=self)
                c.db = self
        else:
            try:
                return self.cluster.cache[type_id, identifier]
            except KeyError:
                logging.warn(
                    f'Component ({type_id}, {identifier}) not found in cache, '
                    'loading from db'
                )
                assert type_id is not None
                assert identifier is not None
                info = self.metadata.get_component(
                    type_id=type_id,
                    identifier=identifier,
                    allow_hidden=allow_hidden,
                )
                c = Document.decode(info, db=self)
                c.db = self

        if c.cache:
            logging.info(f'Adding {c.huuid} to cache')
            self.cluster.cache.put(c)
        return c

    def _add_child_components(self, components, parent, job_events, context):
        # TODO this is a bit of a mess
        # it handles the situation in `Stack` when
        # the components should be added in a certain order
        G = networkx.DiGraph()
        lookup = {(c.type_id, c.identifier): c for c in components}
        for k in lookup:
            G.add_node(k)
            for d in lookup[k].get_children_refs():  # dependencies:
                if d[:2] in lookup:
                    G.add_edge(d, lookup[k].id_tuple)

        nodes = networkx.topological_sort(G)
        create_events = []
        job_events = []
        for n in nodes:
            c, j = self._apply(
                lookup[n], parent=parent.uuid, job_events=job_events, context=context
            )
            create_events += c
            job_events += j
        return create_events, job_events

    def _update_component(self, object, parent: t.Optional[str] = None):
        # TODO add update logic here to check changed attributes
        s.logging.debug(
            f'{object.type_id},{object.identifier} already exists - doing nothing'
        )
        return []

    def _apply(
        self,
        object: Component,
        parent: t.Optional[str] = None,
        artifacts: t.Optional[t.Dict[str, bytes]] = None,
        context: t.Optional[str] = None,
        job_events: t.Sequence['Job'] = (),
    ):
        job_events = list(job_events)

        object.db = self
        existing_versions = self.show(object.type_id, object.identifier)
        already_exists = (
            isinstance(object.version, int) and object.version in existing_versions
        )
        if already_exists:
            self._update_component(object, parent=parent)
            return [], []

        assert hasattr(object, 'identifier')
        assert hasattr(object, 'version')

        if object.version is None:
            if existing_versions:
                object.version = max(existing_versions) + 1
            else:
                object.version = 0

        if object.get_triggers('apply') == ['set_status']:
            object.status = Status.ready

        serialized = object.dict().encode(leaves_to_keep=(Component,))

        for k, v in serialized[KEY_BUILDS].items():
            # TODO this is from the `@component` decorator.
            # Can be handled with a single @leaf decorator around a function
            # or a class
            if isinstance(v, Component) and hasattr(v, 'inline') and v.inline:
                r = dict(v.dict())
                del r['identifier']
                serialized[KEY_BUILDS][k] = r

        children = [
            v for v in serialized[KEY_BUILDS].values() if isinstance(v, Component)
        ]
        create_events, j = self._add_child_components(
            children,
            parent=object,
            job_events=job_events,
            context=context,
        )
        job_events += j
        if children:
            serialized = self._change_component_reference_prefix(serialized)

        serialized = self._save_artifact(object.uuid, serialized)
        if artifacts:
            for file_id, bytes in artifacts.items():
                self.artifact_store.put_bytes(bytes, file_id)

        assert context is not None
        event = Create(
            context=context,
            component=serialized,
            parent=parent,
        )
        create_events.append(event)

        job_events += object.create_jobs(
            event_type='apply', jobs=job_events, context=context
        )
        return create_events, job_events

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
            # TODO - make this less I/O intensive
            component = self.load(
                type_id,
                identifier,
                version=version,
            )
            info = self.metadata.get_component(
                type_id, identifier, version=version, allow_hidden=force
            )
            component.cleanup(self)
            try:
                del self.cluster.cache[component.uuid]
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
        force: bool = False,
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
                    force=force,
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
            self.replace(child, upsert=True, force=force)
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
        self.expire(old_uuid)

    def expire(self, uuid):
        """Expire a component from the cache."""
        self.cluster.cache.expire(uuid)
        parents = self.metadata.get_component_version_parents(uuid)
        while parents:
            for uuid in parents:
                self.cluster.cache.expire(uuid)
            parents = sum(
                [self.metadata.get_component_version_parents(uuid) for uuid in parents],
                [],
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
        logging.info('Getting vector-index')
        vi = self.load('vector_index', vector_index)
        if outputs is None:
            outs: t.Dict = {}
        else:
            outs = outputs.encode()
            if not isinstance(outs, dict):
                raise TypeError(f'Expected dict, got {type(outputs)}')
        logging.info(str(outs))
        return vi.get_nearest(like, db=self, ids=ids, n=n, outputs=outs)

    def disconnect(self):
        """Gracefully shutdown the Datalayer."""
        logging.info("Disconnect from Data Store")
        self.databackend.disconnect()

        logging.info("Disconnect from Metadata Store")
        self.metadata.disconnect()

        logging.info("Disconnect from Artifact Store")
        self.artifact_store.disconnect()

        logging.info("Disconnect from Cluster")
        self.cluster.disconnect()

    def infer_schema(
        self, data: t.Mapping[str, t.Any], identifier: t.Optional[str] = None
    ) -> Schema:
        """Infer a schema from a given data object.

        :param data: The data object
        :param identifier: The identifier for the schema, if None, it will be generated
        :return: The inferred schema
        """
        out = self.databackend.infer_schema(data, identifier)
        return out

    @property
    def cfg(self) -> Config:
        """Get the configuration object for the datalayer."""
        return self._cfg or s.CFG

    @cfg.setter
    def cfg(self, cfg: Config):
        """Set the configuration object for the datalayer."""
        assert isinstance(cfg, Config)
        self._cfg = cfg
