import typing as t
from collections import namedtuple

import click

import superduper as s
from superduper import CFG, logging
from superduper.backends.base.cluster import Cluster
from superduper.backends.base.data_backend import BaseDataBackend
from superduper.base import apply, exceptions
from superduper.base.artifacts import ArtifactStore
from superduper.base.base import Base
from superduper.base.config import Config
from superduper.base.datatype import ComponentType, LeafType
from superduper.base.document import Document
from superduper.base.metadata import (
    MetaDataStore,
    NonExistentMetadataError,
    UniqueConstraintError,
)
from superduper.base.query import Query
from superduper.components.component import Component
from superduper.components.table import Table


# TODO - deprecate hidden logic (not necessary)
class Datalayer:
    """
    Base connector.

    :param databackend: Object containing connection to Datastore.
    :param artifact_store: Object containing connection to Artifactstore.
    :param cluster: Cluster object containing connections to infrastructure.
    """

    def __init__(
        self,
        databackend: BaseDataBackend,
        artifact_store: ArtifactStore,
        cluster: Cluster,
    ):
        """
        Initialize Datalayer.

        :param databackend: Object containing connection to Databackend.
        :param metadata: Object containing connection to Metadatastore.
        :param artifact_store: Object containing connection to Artifactstore.
        :param compute: Object containing connection to ComputeBackend.
        """
        logging.info("Building Data Layer")

        self.artifact_store = artifact_store
        self.artifact_store.db = self

        self.databackend = databackend
        self.databackend.db = self

        self.cluster = cluster
        self.cluster.db = self

        self._cfg = s.CFG
        self.startup_cache: t.Dict[str, t.Any] = {}

        self.metadata = MetaDataStore(self, cache=self.cluster.cache)
        self.metadata.init()

        logging.info("Data Layer built")

    def __getitem__(self, item):
        return Query(table=item, parts=(), db=self)

    def insert(self, items: t.List[Base]):
        """
        Insert data into a table.

        :param items: The instances (`superduper.base.Base`) to insert.
        """
        table = self.pre_insert(items)
        data = [x.dict() for x in items]
        for r in data:
            del r['_path']
        return self[table.identifier].insert(data)

    @property
    def cdc(self):
        """CDC property."""
        return self._cdc

    @cdc.setter
    def cdc(self, cdc):
        """CDC property setter."""
        self._cdc = cdc

    def drop(self, force: bool = False, data: bool = False):
        """
        Drop all data, artifacts, and metadata.

        :param force: Force drop.
        :param data: Drop data.
        """
        if not force and not click.confirm(
            "!!!WARNING USE WITH CAUTION AS YOU WILL"
            "LOSE ALL DATA!!!]\n"
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
            self.databackend.drop_table()

    def show(
        self,
        component: t.Optional[str] = None,
        identifier: t.Optional[str] = None,
        version: t.Optional[int] = None,
        uuid: t.Optional[str] = None,
    ):
        """
        Show available functionality which has been added using ``self.add``.

        If the version is specified, then print full metadata.

        :param component: Component to show ['Model', 'Listener', etc.]
        :param identifier: Identifying string to component.
        :param version: (Optional) Numerical version - specify for full metadata.
        :param uuid: (Optional) UUID of the component.
        """
        if uuid is not None:
            assert component is not None
            return self.metadata.get_component_by_uuid(component, uuid)

        if identifier is None and version is not None:
            raise ValueError(f"Must specify {identifier} to go with {version}")

        if component is None:
            nt = namedtuple('nt', ('component', 'identifier'))
            out = self.metadata.show_components()
            out = sorted(list(set([nt(**x) for x in out])))
            out = [x._asdict() for x in out]
            return out

        if identifier is None:
            try:
                out = self.metadata.show_components(component=component)
            except NonExistentMetadataError:
                return []
            return sorted(out)

        if version is None:
            out = sorted(
                self.metadata.show_component_versions(
                    component=component, identifier=identifier
                )
            )
            return out

        if version == -1:
            return self.metadata.get_component(
                component=component, identifier=identifier, version=None
            )

        return self.metadata.get_component(
            component=component, identifier=identifier, version=version
        )

    def pre_insert(
        self,
        items: t.List[Base],
    ):
        """Pre-insert hook for data insertion.

        :param items: The items to insert.
        """
        table = items[0].__class__.__name__
        try:
            table = self.load('Table', table)
            return table
        except NonExistentMetadataError:
            assert isinstance(items[0], Base)
            return self.metadata.create(type(items[0]))

    def post_insert(self, table: str, ids: t.Sequence[str]):
        """Post-insert hook for data insertion.

        :param table: The table to insert data into.
        :param ids: The IDs of the inserted data.
        """
        if table in self.metadata.show_cdc_tables() and not table.startswith(
            CFG.output_prefix
        ):
            logging.info(f'CDC for {table} is enabled')
            self.cluster.cdc.handle_event(event_type='insert', table=table, ids=ids)

    def _auto_create_table(self, table_name, documents):

        # Should we need to check all the documents?
        document = documents[0]
        table = Table(identifier=table_name, fields=self.infer_schema(document))
        logging.info(f"Creating table {table_name} with schema {table.schema}")
        self.apply(table, force=True)
        return table

    def on_event(self, table: str, ids: t.List[str], event_type: 'str'):
        """
        Trigger computation jobs after data insertion.

        :param table: The table to trigger computation jobs on.
        :param ids: IDs that further reduce the scope of computations.
        :param event_type: The type of event to trigger.
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
        return self.cluster.scheduler.publish(events)  # type: ignore[arg-type]

    def create(self, object: t.Type[Base]):
        """Create a new type of component/ leaf.

        :param object: The object to create.
        """
        try:
            self.metadata.create(object)
        except UniqueConstraintError:
            logging.debug(f'{object} already exists, skipping...')

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
        :param force: Force apply.
        :param wait: Wait for apply events.
        """
        result = apply.apply(db=self, object=object, force=force, wait=wait)
        return result

    def remove(
        self,
        component: str,
        identifier: str,
        version: t.Optional[int] = None,
        recursive: bool = False,
        force: bool = False,
    ):
        """
        Remove a component (version optional).

        :param component: Cmponent to remove ('Model', 'Listener', etc.)
        :param identifier: Identifier of the component (refer to
                            `container.base.Component`).
        :param version: [Optional] Numerical version to remove.
        :param recursive: Toggle to remove all descendants of the component.
        :param force: Force skip confirmation (use with caution).
        """
        if version is not None:
            return self._remove_component_version(
                component, identifier, version=version, force=force, recursive=recursive
            )

        versions = self.metadata.show_component_versions(component, identifier)
        versions_in_use = []
        for v in versions:
            if self.metadata.component_version_has_parents(component, identifier, v):
                versions_in_use.append(v)

        if versions_in_use:
            component_versions_in_use = []
            for v in versions_in_use:
                uuid = self.metadata.get_uuid(component, identifier, v)
                component_versions_in_use.append(
                    f"{uuid} -> "
                    f"{self.metadata.get_component_version_parents(uuid)}",
                )
            if not force:
                raise exceptions.ComponentInUseError(
                    f'Component versions: {component_versions_in_use} are in use'
                )

        if force or click.confirm(
            f'You are about to delete {component}/{identifier}, are you sure?',
            default=False,
        ):
            for v in sorted(list(set(versions) - set(versions_in_use))):
                self._remove_component_version(
                    component, identifier, v, recursive=recursive, force=True
                )

            for v in sorted(versions_in_use):
                self.metadata.hide_component_version(component, identifier, v)

        else:
            logging.warn('aborting.')

    def load(
        self,
        component: str,
        identifier: t.Optional[str] = None,
        version: t.Optional[int] = None,
        uuid: t.Optional[str] = None,
        huuid: t.Optional[str] = None,
        allow_hidden: bool = False,
    ) -> Component:
        """
        Load a component using uniquely identifying information.

        If `uuid` is provided, `component` and `identifier` are ignored.
        If `uuid` is not provided, `component` and `identifier` must be provided.

        :param component: Type ID of the component to load
                         ('datatype', 'model', 'listener', ...).
        :param identifier: Identifier of the component
                           (see `container.base.Component`).
        :param version: [Optional] Numerical version.
        :param uuid: [Optional] UUID of the component to load.
        :param huuid: [Optional] human-readable UUID of the component to load.
        :param allow_hidden: Toggle to ``True`` to allow loading
                             of deprecated components.
        """
        if version is not None:
            assert component is not None
            assert identifier is not None
            info = self.metadata.get_component(
                component=component,
                identifier=identifier,
                version=version,
                allow_hidden=allow_hidden,
            )
            uuid = info['uuid']

        if huuid is not None:
            uuid = huuid.split(':')[-1]

        if uuid is not None:
            info = self.metadata.get_component_by_uuid(
                component=component,
                uuid=uuid,
            )
            # to prevent deserialization propagating back to the cache
            builds = {k: v for k, v in info.get('_builds', {}).items()}
            for k in builds:
                builds[k]['identifier'] = k.split(':')[-1]

            c = LeafType().decode_data(
                {k: v for k, v in info.items() if k != '_builds'},
                builds=builds,
                db=self,
            )
        else:
            assert component is not None
            assert identifier is not None
            logging.info(f'Load ({component, identifier}) from metadata...')
            info = self.metadata.get_component(
                component=component,
                identifier=identifier,
                allow_hidden=allow_hidden,
            )
            c = ComponentType().decode_data(
                info,
                builds=info.get('_builds', {}),
                db=self,
            )
        return c

    def _remove_component_version(
        self,
        component: str,
        identifier: str,
        version: int,
        force: bool = False,
        recursive: bool = False,
    ):
        # TODO - change this logic for a not version-by-version deletion
        if version is None:
            return

        try:
            r = self.metadata.get_component(component, identifier, version=version)
        except NonExistentMetadataError:
            logging.warn(
                f'Component {component}:{identifier}:{version} has already been removed'
            )
            return

        if self.metadata.component_version_has_parents(component, identifier, version):
            parents = self.metadata.get_component_version_parents(r['uuid'])
            raise exceptions.ComponentInUseError(
                f'{r["uuid"]} is involved in other components: {parents}'
            )

        if not (
            force
            or click.confirm(
                f'You are about to delete {component}/{identifier}{version}, '
                'are you sure?',
                default=False,
            )
        ):
            return

        object = self.load(
            component,
            identifier,
            version=version,
        )
        info = self.metadata.get_component(
            component, identifier, version=version, allow_hidden=force
        )
        object.cleanup(self)
        self._delete_artifacts(r['uuid'], info)
        self.metadata.delete_component_version(component, identifier, version=version)
        self.metadata.delete_parent_child_relationships(r['uuid'])

        if not recursive:
            return

        children = object.get_children(deep=True)

        children = object.sort_components(children)[::-1]
        for c in children:
            if c.version is None:
                logging.warn(f'Found uninitialized component {c.huuid}, skipping...')
                continue
            try:
                self._remove_component_version(
                    c.component,
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
        """
        assert object.version is not None

        old_uuid = None
        try:
            info = self.metadata.get_component(
                object.__class__.__name__, object.identifier, version=object.version
            )
            old_uuid = info['uuid']
        except NonExistentMetadataError:
            pass

        serialized = object.dict()

        if old_uuid:

            def _replace_fn(component):
                self.replace(component)
                return f'&:component:{component.huuid}'

            serialized = serialized.map(_replace_fn, lambda x: isinstance(x, Component))
            serialized = serialized.encode(keep_schema=False)

            self._delete_artifacts(object.uuid, info)
            artifact_ids, _ = self._find_artifacts(info)
            self.metadata.create_artifact_relation(object.uuid, artifact_ids)
            serialized = self._save_artifact(object.uuid, serialized)

            if old_uuid == object.uuid:
                self.metadata.replace_object(
                    object.__class__.__name__, object.uuid, serialized
                )
            else:
                self.metadata.create_component(serialized, path=serialized['_path'])
                self.metadata.remove_by_uuid(info['_path'].split('.')[-1], old_uuid)

        else:
            serialized = serialized.encode(keep_schema=False)
            self.metadata.create_component(serialized)

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
        vi = self.load('VectorIndex', vector_index)
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
        logging.info("Disconnect from Cluster")
        self.cluster.disconnect()

    @property
    def cfg(self) -> Config:
        """Get the configuration object for the datalayer."""
        return self._cfg or s.CFG

    @cfg.setter
    def cfg(self, cfg: Config):
        """Set the configuration object for the datalayer."""
        assert isinstance(cfg, Config)
        self._cfg = cfg

    def execute(self, query: str):
        """Execute a native DB query.

        :param query: The query to execute.
        """
        return self.databackend.execute_native(query)
