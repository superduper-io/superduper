import dataclasses as dc
import random
import typing as t
import warnings
from collections import namedtuple

import click
import networkx

import superduper as s
from superduper import CFG, logging
from superduper.backends.base.artifacts import ArtifactStore
from superduper.backends.base.backends import vector_searcher_implementations
from superduper.backends.base.cluster import Cluster
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
from superduper.base.event import ComponentPlaceholder, Event, EventType
from superduper.components.component import Component
from superduper.components.datatype import DataType
from superduper.components.schema import Schema
from superduper.components.table import Table
from superduper.misc.annotations import deprecated
from superduper.misc.colors import Colors
from superduper.misc.download import download_from_one
from superduper.misc.retry import db_retry
from superduper.misc.special_dicts import recursive_update
from superduper.vector_search.base import BaseVectorSearcher

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
        if refresh:
            self.cluster.cdc.handle_event(event_type=EventType.delete, query=delete, ids=result)
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

        logging.info(f'Inserted {len(inserted_ids)} documents into {insert.table}')
        logging.debug(f'Inserted IDs: {inserted_ids}')

        # Do we want to hide this
        self.cluster.cdc.handle_event(event_type=EventType.insert, query=insert, ids=inserted_ids)

        # if refresh:
            # if cdc_status and not is_output_table:
            #     logging.warn('CDC service is active, skipping model/listener refresh')
            # else:
            #     return inserted_ids, self.on_event(
            #         insert,
            #         ids=inserted_ids,
            #     )

        return inserted_ids, None

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
            dest = ComponentPlaceholder(identifier=identifier, type_id=type_id)
            events.extend(
                [Event(dest=dest, ids=[str(id)], event_type=event_type) for id in ids]
            )

        logging.info(
            f'Created {len(events)} events for {event_type} on [{query.table}]'
        )
        logging.info(f'Publishing {len(events)} events')
        return self.cluster.queue.publish(events)

    def _write(self, write: Query, refresh: bool = True) -> UpdateResult:
        """
        Bulk write data to the database.

        :param write: The update query object specifying the data to be written.
        :param refresh: Boolean indicating whether to refresh the task group on write.
        """
        _, updated_ids, deleted_ids = write.do_execute(self)

        if refresh:

            if updated_ids:
                self.cluster.cdc.handle_event(event_type=EventType.update, query=write, ids=updated_ids)

            if deleted_ids:
                self.cluster.cdc.handle_event(event_type=EventType.delete, query=write, ids=deleted_ids)
                
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

        # cdc_status = s.CFG.cluster.cdc.uri is not None
        # is_output_table = update.table.startswith(CFG.output_prefix)

        if refresh:
            self.cluster.cdc.handle_event(event_type=EventType.update, query=update, ids=updated_ids)

        # if refresh and updated_ids:
        #     if cdc_status and not is_output_table:
        #         logging.warn('CDC service is active, skipping model/listener refresh')
        #     else:
        #         # Overwrite should be true since updates could be done on collections
        #         # with already existing outputs.
        #         # We need overwrite outputs on those select and recompute predict
        #         return updated_ids, self.on_event(
        #             query=update, ids=updated_ids, event_type=EventType.upsert
        #         )
        return updated_ids, None

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
        return self._apply(object=object), object

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

        if info.get('cache', False):
            try:
                return self.cluster.cache[info['type_id'], info['identifier']]
            except KeyError:
                logging.info(f'Component {info["uuid"]} not found in cache, loading from db')

        m = Document.decode(info, db=self)
        m.db = self
        m.on_load(self)

        assert type_id is not None
        if m.cache:
            logging.info(f'Adding component {info["uuid"]} to cache.')
            self.cluster.cache.put(m)
        return m

    def _add_child_components(self, components, parent):
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
        for n in nodes:
            self._apply(lookup[n], parent=parent.uuid)

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
    ):
        object.db = self
        object.pre_create(self)
        assert hasattr(object, 'identifier')
        assert hasattr(object, 'version')

        existing_versions = self.show(object.type_id, object.identifier)

        already_exists = (
            isinstance(object.version, int) and object.version in existing_versions
        )

        if already_exists:
            return self._update_component(object, parent=parent)

        if object.version is None:
            if existing_versions:
                object.version = max(existing_versions) + 1
            else:
                object.version = 0

        serialized = object.dict().encode(leaves_to_keep=(Component,))

        for k, v in serialized[KEY_BUILDS].items():
            # TODO What is this "inline"?
            if isinstance(v, Component) and hasattr(v, 'inline') and v.inline:
                r = dict(v.dict())
                del r['identifier']
                serialized[KEY_BUILDS][k] = r

        children = [
            v for v in serialized[KEY_BUILDS].values() if isinstance(v, Component)
        ]
        self._add_child_components(
            children,
            parent=object,
        )
        if children:
            serialized = self._change_component_reference_prefix(serialized)

        serialized = self.artifact_store.save_artifact(serialized)
        if artifacts:
            for file_id, bytes in artifacts.items():
                self.artifact_store.put_bytes(bytes, file_id)

        self.metadata.create_component(serialized)

        if parent is not None:
            self.metadata.create_parent_child(parent, object.uuid)

        object.post_create(self)

        event = Event(
            event_type='apply',
            dest=ComponentPlaceholder(
                type_id=object.type_id, identifier=object.identifier
            ),
        )
        self.cluster.queue.publish([event])

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
                type_id, identifier, version=version, allow_hidden=force
            )
            info = self.metadata.get_component(
                type_id, identifier, version=version, allow_hidden=force
            )
            component.cleanup(self)
            try:
                del self.cluster.cache[component.uuid]
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

        self.artifact_store.delete_artifact(info)

        serialized = self.artifact_store.save_artifact(serialized)

        self.metadata.replace_object(
            serialized,
            identifier=object.identifier,
            type_id=object.type_id,
            version=object.version,
        )
        self.expire(object.uuid)

    def expire(self, uuid):
        parents = True
        self.cluster.cache.expire(uuid)
        parents = self.metadata.get_component_version_parents(uuid)
        while parents:
            for uuid in parents:
                self.cluster.cache.expire(uuid)
            parents = sum(
                [self.metadata.get_component_version_parents(uuid) for uuid in parents],
                [],
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