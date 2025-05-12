import json
import time
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
from superduper.base.datatype import BaseType, ComponentType
from superduper.base.document import Document
from superduper.base.event import Delete
from superduper.base.metadata import (
    JOB_PHASE_FAILED,
    JOB_PHASE_RUNNING,
    MetaDataStore,
)
from superduper.base.query import Query
from superduper.components.component import Component
from superduper.components.table import Table
from superduper.misc.importing import isreallyinstance


# TODO - deprecate hidden logic (not necessary)
class Datalayer:
    """
    Base connector.

    :param databackend: Object containing connection to Datastore.
    :param artifact_store: Object containing connection to Artifactstore.
    :param cluster: Cluster object containing connections to infrastructure.
    :param metadata: Object containing connection to Metadatastore.
    """

    def __init__(
        self,
        databackend: BaseDataBackend,
        artifact_store: ArtifactStore,
        cluster: Cluster | None,
        metadata: BaseDataBackend | None,
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

        self.databackend: BaseDataBackend = databackend
        self.databackend.db = self

        self.cluster = cluster
        if self.cluster:
            self.cluster.db = self

        self._cfg = s.CFG
        self.startup_cache: t.Dict[str, t.Any] = {}
        self._component_cache: t.Dict[t.Tuple[str, str], Component] = {}

        if metadata:
            self.metadata = MetaDataStore(metadata, parent_db=self)  # type: ignore[arg-type]
        else:
            self.metadata = MetaDataStore(self, parent_db=self)
        self.metadata.init()

        logging.info("Data Layer built")

    def __getitem__(self, item):
        return Query(table=item, parts=(), db=self)

    def initialize(self):
        """Initialize the Datalayer."""
        if self.cluster:
            self.cluster.initialize()

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

    # Add option to drop a whole class of components
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
        if self.cluster:
            self.cluster.drop(force=True)

        self.databackend.drop(force=True)
        if self.artifact_store:
            self.artifact_store.drop(force=True)
        self.metadata.drop(force=True)

    def wait(
        self,
        component: str,
        identifier: str,
        uuid: str | None = None,
        heartbeat: float = 1.0,
        timeout: int = 30,
    ) -> None:
        """
        Wait for a component to be ready.

        :param component: Component to wait for.
        :param identifier: Identifier of the component.
        :param uuid: UUID of the component (optional).
        :param heartbeat: Time between status checks in seconds
        :param timeout: Maximum wait time in seconds

        :raises TimeoutError: If the component doesn't become ready within the timeout period.
        :raises InternalServerError: If the component enters a 'failed' state.
        """
        start = time.time()
        component_id = f"{component}:{identifier}"

        while True:
            # Check for timeout at the beginning of each iteration
            if time.time() - start > timeout:
                raise exceptions.TimeoutError(
                    f'Timed out waiting for component to become {JOB_PHASE_RUNNING}.'
                )

            try:
                # Get component based on uuid or identifier
                if uuid:
                    r = self.metadata.get_component_by_uuid(
                        component=component, uuid=uuid
                    )
                else:
                    r = self.metadata.get_component(
                        component=component, identifier=identifier
                    )

                # Parse status
                status = r['status']
                if not CFG.json_native:
                    status = json.loads(status)

                # Check the phase of the object
                if status['phase'] == JOB_PHASE_RUNNING:
                    # object is running. return immediately.
                    logging.info(f"{component}:{identifier} is running")
                    return
                elif status['phase'] == JOB_PHASE_FAILED:
                    # object has failed. throw an error.
                    err_msg = f"{component_id} failed with status {status}"
                    raise exceptions.InternalError(err_msg, None)
                else:
                    # object found, but has not reached desired state.
                    logging.info(
                        f"{component_id} is not ready yet with status {status}"
                    )

            # Object not found
            except exceptions.NotFound:
                logging.info(f"Component {component_id} cannot be found. Retry...")

            # Wait before checking again, regardless of exception or status
            time.sleep(heartbeat)

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
                out = self.metadata.show_components(
                    component=component,
                )
            except exceptions.NotFound:
                return []
            return sorted(out)

        if version is None:
            try:
                out = sorted(
                    self.metadata.show_component_versions(
                        component=component, identifier=identifier
                    )
                )
                return out
            except exceptions.NotFound:
                return []

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
        except exceptions.NotFound:
            assert isreallyinstance(items[0], Base)
            return self.metadata.create(type(items[0]))

    def _post_query(self, table: str, ids: t.Sequence[str], type_: str):
        if table in self.metadata.show_cdc_tables() and not table.startswith(
            CFG.output_prefix
        ):
            logging.info(f'CDC for {table} is enabled')
            assert self.cluster is not None
            self.cluster.cdc.handle_event(event_type=type_, table=table, ids=ids)

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
        assert self.cluster is not None
        return self.cluster.scheduler.publish(events)  # type: ignore[arg-type]

    def create(self, object: t.Type[Base]):
        """Create a new type of component/ leaf.

        :param object: The object to create.
        """
        try:
            self.metadata.create(object)
        except exceptions.AlreadyExists:
            logging.debug(f'{object} already exists, skipping...')

    def apply(
        self,
        object: t.Union[Component, t.Sequence[t.Any], t.Any],
        force: bool | None = None,
        wait: bool = False,
        jobs: bool = True,
    ):
        """
        Add functionality in the form of components.

        Components are stored in the configured artifact store
        and linked to the primary database through metadata.

        :param object: Object to be stored.
        :param force: Force apply.
        :param wait: Wait for apply events.
        :param jobs: Execute jobs.
        """
        result = apply.apply(db=self, object=object, force=force, wait=wait, jobs=jobs)
        return result

    def remove(
        self,
        component: str,
        identifier: str,
        recursive: bool = False,
        force: bool = False,
    ):
        """
        Remove a component (version optional).

        :param component: Cmponent to remove ('Model', 'Listener', etc.)
        :param identifier: Identifier of the component (refer to
                            `container.base.Component`).
        :param recursive: Toggle to remove all descendants of the component.
        :param force: Toggle to force remove the component.
        """
        events: t.List[Delete] = []
        failed: t.List[str] = []
        self._build_remove(
            component=component,
            identifier=identifier,
            events=events,
            failed=failed,
            recursive=recursive,
        )

        if failed and not force:
            raise exceptions.Conflict(
                component, identifier, f"the following components are in use: {failed}"
            )

        for i, e in enumerate(events):
            logging.info(
                f'Removing component [{i + 1}/{len(events)}] '
                f'{e.component}:{e.identifier}'
            )
            e.execute(self)
            logging.info(
                f'Removing component [{i + 1}/{len(events)}] '
                f'{e.component}:{e.identifier}... DONE'
            )

    def _build_remove(
        self,
        component: str,
        identifier: str,
        events: t.List,
        failed: t.List,
        recursive: bool = False,
    ):

        object = self.load(component=component, identifier=identifier)

        previous = [e.huuid for e in events]

        parents = self.metadata.get_component_parents(
            component=component, identifier=identifier
        )
        fail = False
        if parents:
            # Only fail the deletion attempt if the parents aren't in this cascade
            for p in parents:
                if f'{p[0]}:{p[1]}' not in previous:
                    failed.append(f'{component}:{identifier} -> {p[0]}:{p[1]}')
                    fail = True

            # If the deletion fails, we need to stop
            if fail:
                return

        events.append(Delete(component=component, identifier=identifier))

        if recursive:
            children = object.get_children()
            for c in children:
                self._build_remove(
                    c.component,
                    c.identifier,
                    recursive=True,
                    events=events,
                    failed=failed,
                )

    def load_all(self, component: str, **kwargs) -> t.List[Component]:
        """Load all instances of component.

        :param component: Component class
        :param kwargs: Addition key-value pairs to `self.load`
        """
        identifiers = self.metadata.show_components(component=component)
        out: t.List[Component] = []
        for identifier in identifiers:
            try:
                c = self.load(component, identifier)
                applies = True
                for k, v in kwargs.items():
                    if getattr(c, k) != v:
                        applies = False
                        break
                if applies:
                    out.append(c)
            except exceptions.NotFound:
                continue
        return out

    def load(
        self,
        component: str,
        identifier: t.Optional[str] = None,
        version: t.Optional[int] = None,
        uuid: t.Optional[str] = None,
        huuid: t.Optional[str] = None,
        overrides: t.Dict | None = None,
        component_cache: bool = True,
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
        :param overrides: [Optional] Dictionary of overrides to apply to the component.
        :param component_cache: [Optional] Whether to use the component cache.
        """
        if version is not None:
            assert component is not None
            assert identifier is not None
            info = self.metadata.get_component(
                component=component,
                identifier=identifier,
                version=version,
            )
            uuid = info['uuid']
            info.update(overrides or {})
        else:
            if component_cache and (component, identifier) in self._component_cache:

                assert isinstance(identifier, str)
                if self._component_cache[
                    (component, identifier)
                ].uuid == self.metadata.get_latest_uuid(
                    component=component,
                    identifier=identifier,
                ):
                    logging.info(f'Found {component, identifier} in cache...')
                    return self._component_cache[(component, identifier)]
                else:
                    logging.info(
                        f'Found {component, identifier} '
                        'in cache but UUID does not match...'
                    )
                    del self._component_cache[(component, identifier)]

        if huuid is not None:
            uuid = huuid.split(':')[-1]

        if uuid is not None:
            info = self.metadata.get_component_by_uuid(
                component=component,
                uuid=uuid,
            )
            info.update(overrides or {})

            # to prevent deserialization propagating back to the cache
            builds = {k: v for k, v in info.get('_builds', {}).items()}
            for k in builds:
                builds[k]['identifier'] = k.split(':')[-1]

            c = BaseType().decode_data(
                {k: v for k, v in info.items() if k != '_builds'},
                builds=builds,
                db=self,
            )
        elif identifier is not None:
            assert component is not None
            logging.debug(f'Load ({component, identifier}) from metadata...')
            info = self.metadata.get_component(
                component=component,
                identifier=identifier,
            )
            info.update(overrides or {})
            c = ComponentType().decode_data(
                info,
                builds=info.get('_builds', {}),
                db=self,
            )
        else:
            raise ValueError(
                'Must provide either `uuid` or `component` and `identifier`'
            )

        if getattr(c, 'component_cache', False):
            self._component_cache[(c.component, c.identifier)] = c
        return c

    def _save_artifact(self, info: t.Dict):
        """
        Save an artifact to the artifact store.

        :param artifact: The artifact to save.
        """
        return self.artifact_store.save_artifact(info)

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
        vector_index = self.load('VectorIndex', vector_index)
        if outputs is None:
            outs: t.Dict = {}
        else:
            outs = outputs.encode()
            if not isinstance(outs, dict):
                raise TypeError(f'Expected dict, got {type(outputs)}')
        logging.info(str(outs))
        return vector_index.get_nearest(like, ids=ids, n=n, outputs=outs)

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
        assert isreallyinstance(cfg, Config)
        self._cfg = cfg

    def execute(self, query: str):
        """Execute a native DB query.

        :param query: The query to execute.
        """
        return self.databackend.execute_native(query)
