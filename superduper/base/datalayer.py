import random
import typing as t
from collections import namedtuple

import click

import superduper as s
from superduper import CFG, logging
from superduper.backends.base.artifacts import ArtifactStore
from superduper.backends.base.cluster import Cluster
from superduper.backends.base.compute import ComputeBackend
from superduper.backends.base.data_backend import BaseDataBackend
from superduper.backends.base.metadata import MetaDataStore
from superduper.backends.base.query import Query
from superduper.backends.local.cluster import LocalCluster
from superduper.base import apply, exceptions
from superduper.base.config import Config
from superduper.base.cursor import SuperDuperCursor
from superduper.base.document import Document
from superduper.components.component import Component
from superduper.components.datatype import BaseDataType
from superduper.components.schema import Schema
from superduper.components.table import Table
from superduper.misc.annotations import deprecated
from superduper.misc.colors import Colors
from superduper.misc.importing import import_object
from superduper.misc.retry import db_retry

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
        logging.info("Data Layer built")

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
            out = [x._asdict() for x in out]
            return out

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
        datatypes: t.Sequence[BaseDataType] = (),
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

        if not insert.documents:
            logging.info(f'No documents to insert into {insert.table}')
            return []

        for r in insert.documents:
            r.setdefault(
                '_fold',
                'train' if random.random() >= s.CFG.fold_probability else 'valid',
            )
        if auto_schema and self.cfg.auto_schema:
            schema = self._auto_create_table(insert.table, insert.documents).schema

            timeout = 60

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
                    time.sleep(1)
                    continue
                break

            if not exists:
                raise TimeoutError(
                    f'{insert.table} not found after {timeout} seconds'
                    ' table auto creation likely has failed or is stalling...'
                )
            for r in insert.documents:
                r.schema = schema

        inserted_ids = insert.do_execute(self)

        logging.info(f'Inserted {len(inserted_ids)} documents into {insert.table}')
        logging.debug(f'Inserted IDs: {inserted_ids}')

        if not refresh:
            return []

        if (
            isinstance(self.cluster, LocalCluster)
            and insert.table in self.metadata.show_cdc_tables()
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
        return table

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

        if not self.metadata.show_cdcs(table):
            logging.info(
                f'Skipping cdc for inserted documents in ({table})',
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

        if (
            isinstance(self.cluster, LocalCluster)
            and table in self.metadata.show_cdc_tables()
            and not table.startswith(CFG.output_prefix)
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
        wait: bool = False,
    ):
        """
        Add functionality in the form of components.

        Components are stored in the configured artifact store
        and linked to the primary database through metadata.

        :param object: Object to be stored.
        :param dependencies: List of jobs which should execute before component
                             initialization begins.
        :param wait: Wait for apply events.
        :return: Tuple containing the added object(s) and the original object(s).
        """
        result = apply.apply(db=self, object=object, force=force, wait=wait)
        return result

    def remove(
        self,
        type_id: str,
        identifier: str,
        version: t.Optional[int] = None,
        recursive: bool = False,
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
                type_id, identifier, version=version, force=force, recursive=recursive
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

        if force or click.confirm(
            f'You are about to delete {type_id}/{identifier}, are you sure?',
            default=False,
        ):
            for v in sorted(list(set(versions) - set(versions_in_use))):
                self._remove_component_version(
                    type_id, identifier, v, recursive=recursive, force=True
                )

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
        :param huuid: [Optional] human-readable UUID of the component to load.
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
                logging.info(
                    f'Component {uuid} not found in cache, loading from db with uuid'
                )
                info = self.metadata.get_component_by_uuid(
                    uuid=uuid, allow_hidden=allow_hidden
                )
                try:
                    class_schema = import_object(info['_path']).build_class_schema()
                except (KeyError, AttributeError):
                    # if defined in __main__ then the class is directly serialized
                    assert '_object' in info
                    from superduper.components.datatype import DEFAULT_SERIALIZER, Blob

                    bytes_ = Blob(
                        identifier=info['_object'].split(':')[-1], db=self
                    ).unpack()
                    object = DEFAULT_SERIALIZER.decode_data(bytes_)
                    class_schema = object.build_class_schema()

                c = Document.decode(info, db=self, schema=class_schema)
                c.db = self
                if c.cache:
                    logging.info(f'Adding {c.huuid} to cache')
                    self.cluster.cache.put(c)
        else:
            try:
                c = self.cluster.cache[type_id, identifier]
                logging.debug(
                    f'Component {(type_id, identifier)} was found in cache...'
                )
            except KeyError:
                logging.info(
                    f'Component ({type_id}, {identifier}) not found in cache, '
                    'loading from db'
                )
                assert type_id is not None
                assert identifier is not None
                logging.info(f'Load ({type_id, identifier}) from metadata...')
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

    def _remove_component_version(
        self,
        type_id: str,
        identifier: str,
        version: int,
        force: bool = False,
        recursive: bool = False,
    ):
        try:
            r = self.metadata.get_component(type_id, identifier, version=version)
        except FileNotFoundError:
            logging.warn(
                f'Component {type_id}:{identifier}:{version} has already been removed'
            )
            return
        if self.metadata.component_version_has_parents(type_id, identifier, version):
            parents = self.metadata.get_component_version_parents(r['uuid'])
            raise exceptions.ComponentInUseError(
                f'{r["uuid"]} is involved in other components: {parents}'
            )

        if not (
            force
            or click.confirm(
                f'You are about to delete {type_id}/{identifier}{version}, '
                'are you sure?',
                default=False,
            )
        ):
            return

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

        if not recursive:
            return

        children = component.get_children(deep=True)
        children = component.sort_components(children)[::-1]
        for c in children:
            self.metadata.delete_parent_child(component.uuid, c.uuid)
            assert isinstance(c.version, int)
            try:
                self._remove_component_version(
                    c.type_id,
                    c.identifier,
                    version=c.version,
                    recursive=False,
                    force=force,
                )
            except exceptions.ComponentInUseError as e:
                if force:
                    logging.warn(
                        f'Component {c.huuid} is in use: {e}\n'
                        'Skipping since force=True...'
                    )
                else:
                    raise e

    def replace(self, object: t.Any):
        """
        Replace a model in the artifact store with an updated object.

        (Use with caution!)

        :param object: The object to replace.
        :param upsert: Toggle to ``True`` to enable replacement even if
                       the object doesn't exist yet.
        :param force: set to `True` to skip confirm # TODO
        """
        old_uuid = None
        try:
            info = self.metadata.get_component(
                object.type_id, object.identifier, version=object.version
            )
            old_uuid = info['uuid']
        except FileNotFoundError:
            pass

        serialized = object.dict()

        if old_uuid:

            def _replace_fn(component):
                if getattr(component, 'inline', False):
                    return component
                self.replace(component)
                return f'&:component:{component.huuid}'

            serialized = serialized.map(_replace_fn, lambda x: isinstance(x, Component))
            serialized = serialized.encode(keep_schema=False)

            self._delete_artifacts(object.uuid, info)
            artifact_ids, _ = self._find_artifacts(info)
            self.metadata.create_artifact_relation(object.uuid, artifact_ids)
            serialized = self._save_artifact(object.uuid, serialized)

            self.metadata.replace_object(
                serialized,
                identifier=object.identifier,
                type_id=object.type_id,
                version=object.version,
            )
            self.expire(old_uuid)
            self._remove_old_children_relations(object)
        else:
            serialized = serialized.encode(keep_schema=False)
            self.metadata.create_component(serialized)

    def _remove_old_children_relations(self, object: Component):
        exists_relations = self.metadata.get_children_relations(object.uuid)
        now_relations = [c.uuid for c in object.get_children(deep=True)]
        delete_relations = set(exists_relations) - set(now_relations)
        for c in delete_relations:
            self.metadata.delete_parent_child(object.uuid, c)

    def expire(self, uuid):
        """Expire a component from the cache."""
        self.cluster.cache.expire(uuid)
        self.metadata.expire(uuid)
        parents = self.metadata.get_component_version_parents(uuid)
        for parent in parents:
            self.expire(parent)

    def _save_artifact(self, uuid, info: t.Dict):
        """
        Save an artifact to the artifact store.

        :param artifact: The artifact to save.
        """
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
        # TODO deprecate
        # like = self._get_content_for_filter(like)
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
