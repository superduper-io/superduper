"""
Datalayer module for superduper.io.

This module provides the main interface for interacting with data storage,
artifacts, and cluster management in the superduper system.
"""

import typing as t
from collections import namedtuple
from pathlib import Path

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
from superduper.base.metadata import (
    MetaDataStore,
    NonExistentMetadataError,
    UniqueConstraintError,
)
from superduper.base.query import Query
from superduper.components.component import Component
from superduper.components.table import Table
from superduper.misc.importing import isreallyinstance
from superduper.misc.special_dicts import recursive_find


class Datalayer:
    """
    Main data connector for superduper.io.

    The Datalayer connects the various storage and processing components:
    - Data backend for storing structured data
    - Artifact store for binary/file data
    - Cluster for distributed computing
    - Metadata store for component registration and discovery
    """

    def __init__(
            self,
            databackend: BaseDataBackend,
            artifact_store: ArtifactStore,
            cluster: Cluster,
    ):
        """
        Initialize Datalayer with storage backends and cluster.

        Args:
            databackend: Connection to the data storage system
            artifact_store: Connection to the artifact storage system
            cluster: Connection to cluster infrastructure
        """
        logging.info("Building Data Layer")

        # Initialize storage components
        self.artifact_store = artifact_store
        self.artifact_store.db = self

        self.databackend = databackend
        self.databackend.db = self

        self.cluster = cluster
        self.cluster.db = self

        # Initialize configuration
        self._cfg = s.CFG
        self.startup_cache = {}

        # Initialize metadata store
        self.metadata = MetaDataStore(self, cache=self.cluster.cache)
        self.metadata.init()

        logging.info("Data Layer built")

    def __getitem__(self, item: str) -> Query:
        """
        Access a table by name using dictionary-style access.

        Args:
            item: Table name to access

        Returns:
            Query object for the specified table
        """
        return Query(table=item, parts=(), db=self)

    def insert(self, items: t.List[Base]) -> t.List[str]:
        """
        Insert data objects into their appropriate table.

        Args:
            items: List of Base instances to insert

        Returns:
            List of inserted object IDs
        """
        # Validate input
        if len(items) == 0:
            raise ValueError("Cannot insert empty list of items")

        # Get table from first item's class name
        table_name = items[0].__class__.__name__

        # Ensure all items are Base instances
        if not isreallyinstance(items[0], Base):
            raise AssertionError("Items must be Base instances")

        # Try to load existing table
        table = None
        try:
            table = self.load('Table', table_name)
        except NonExistentMetadataError:
            # Create table if it doesn't exist
            table = self.metadata.create(type(items[0]))

        # Prepare data for insertion
        data = []
        for item in items:
            item_dict = item.dict()
            del item_dict['_path']
            data.append(item_dict)

        # Insert data
        ids = self[table.identifier].insert(data)

        # Handle CDC if enabled
        if table.identifier in self.metadata.show_cdc_tables():
            if not table.identifier.startswith(CFG.output_prefix):
                logging.info(f'CDC for {table.identifier} is enabled')
                self.cluster.cdc.handle_event(
                    event_type='insert',
                    table=table.identifier,
                    ids=ids
                )

        return ids

    def drop(self, force: bool = False, data: bool = False) -> None:
        """
        Drop database components (metadata, artifacts, etc).

        Args:
            force: Skip confirmation prompt when True
            data: Also drop data in addition to schema when True
        """
        if not force:
            confirmed = click.confirm(
                "!!! WARNING: USE WITH CAUTION - YOU WILL LOSE ALL DATA !!!\n"
                "Are you sure you want to drop the database?",
                default=False,
            )
            if not confirmed:
                logging.warn("Aborting...")
                return

        # Drop cluster components (cache, vector-indexes, triggers, queues)
        self.cluster.drop(force=True)

        if data:
            # Drop all data and artifacts
            self.databackend.drop(force=True)
            self.artifact_store.drop(force=True)
        else:
            # Only drop table structure
            self.databackend.drop_table()

    def show(
            self,
            component: t.Optional[str] = None,
            identifier: t.Optional[str] = None,
            version: t.Optional[int] = None,
            uuid: t.Optional[str] = None,
    ) -> t.Union[t.List[t.Dict], t.Dict, t.List[str]]:
        """
        Show available components registered in the system.

        Args:
            component: Component type to filter by ('Model', 'Listener', etc.)
            identifier: Specific component identifier to show
            version: Specific version number to show detailed metadata
            uuid: Specific component UUID to retrieve

        Returns:
            List of components or detailed component information
        """
        # Case 1: Query by UUID
        if uuid is not None:
            if component is None:
                raise ValueError("Must specify component when using uuid")
            return self.metadata.get_component_by_uuid(component, uuid)

        # Case 2: Validate parameters
        if identifier is None and version is not None:
            raise ValueError(f"Must specify identifier to go with version {version}")

        # Case 3: Show all components
        if component is None:
            nt = namedtuple('nt', ('component', 'identifier'))
            out = self.metadata.show_components()
            result = []
            for item in out:
                component_obj = nt(**item)
                result.append(component_obj)
            result = sorted(list(set(result)))

            final_result = []
            for item in result:
                final_result.append(item._asdict())
            return final_result

        # Case 4: Show components of a specific type
        if identifier is None:
            try:
                out = self.metadata.show_components(component=component)
                return sorted(out)
            except NonExistentMetadataError:
                return []

        # Case 5: Show versions or specific version
        if version is None:
            # Show all versions
            versions = self.metadata.show_component_versions(
                component=component, identifier=identifier
            )
            return sorted(versions)
        elif version == -1:
            # Show latest version
            return self.metadata.get_component(
                component=component, identifier=identifier, version=None
            )
        else:
            # Show specific version
            return self.metadata.get_component(
                component=component, identifier=identifier, version=version
            )

    def on_event(self, table: str, ids: t.List[str], event_type: str) -> t.Any:
        """
        Trigger computation jobs after data changes.

        Args:
            table: Table where data changed
            ids: IDs of affected records
            event_type: Type of event ('insert', 'update', 'delete')

        Returns:
            Result from publishing events to scheduler
        """
        from superduper.base.event import Change

        # Check if table has CDC consumers
        has_cdcs = self.metadata.show_cdcs(table)
        if not has_cdcs:
            logging.info(
                f'Skipping CDC for {event_type} events in {table} '
                'because no component is listening to this table.'
            )
            return None

        # Create events for each affected ID
        events = []
        for id in ids:
            event = Change(ids=[str(id)], queue=table, type=event_type)
            events.append(event)

        logging.info(f'Created {len(events)} events for {event_type} on [{table}]')
        logging.info(f'Publishing {len(events)} events')
        return self.cluster.scheduler.publish(events)

    def create(self, obj_type: t.Type[Base]) -> None:
        """
        Register a new component type in the system.

        Args:
            obj_type: The type to register
        """
        try:
            self.metadata.create(obj_type)
        except UniqueConstraintError:
            logging.debug(f'{obj_type.__name__} already exists, skipping...')

    def apply(
            self,
            object: t.Union[Component, t.Sequence[t.Any], t.Any],
            force: t.Optional[bool] = None,
            wait: bool = False,
            jobs: bool = True,
    ) -> t.Any:
        """
        Register and store a component in the system.

        Args:
            object: Component(s) to register
            force: Force apply if True, even if component exists
            wait: Wait for apply events to complete if True
            jobs: Execute related jobs if True

        Returns:
            Result of the apply operation
        """
        return apply.apply(db=self, object=object, force=force, wait=wait, jobs=jobs)

    def remove(
            self,
            component: str,
            identifier: str,
            version: t.Optional[int] = None,
            recursive: bool = False,
            force: bool = False,
    ) -> None:
        """
        Remove a component from the system.

        Args:
            component: Component type to remove
            identifier: Component identifier
            version: Specific version to remove (all versions if None)
            recursive: Also remove child components if True
            force: Skip confirmation and dependency checks if True
        """
        # Remove a specific version if provided
        if version is not None:
            self._remove_single_version(
                component, identifier, version, recursive, force
            )
            return

        # Get all versions of the component
        versions = self.metadata.show_component_versions(component, identifier)

        # Find versions that are in use by other components
        versions_in_use = []
        for v in versions:
            if self.metadata.component_version_has_parents(component, identifier, v):
                versions_in_use.append(v)

        # Check if any version is in use
        if len(versions_in_use) > 0 and not force:
            component_versions_in_use = []
            for v in versions_in_use:
                uuid = self.metadata.get_uuid(component, identifier, v)
                parents = self.metadata.get_component_version_parents(uuid)
                component_versions_in_use.append(f"{uuid} -> {parents}")

            raise exceptions.ComponentInUseError(
                f'Component versions: {component_versions_in_use} are in use'
            )

        # Confirm deletion
        should_proceed = False
        if force:
            should_proceed = True
        else:
            should_proceed = click.confirm(
                f'You are about to delete {component}/{identifier}, are you sure?',
                default=False,
            )

        if not should_proceed:
            logging.warn('Aborting.')
            return

        # Remove unused versions
        unused_versions = []
        for v in versions:
            if v not in versions_in_use:
                unused_versions.append(v)

        for v in sorted(unused_versions):
            self._remove_single_version(
                component, identifier, v, recursive=recursive, force=True
            )

        # Hide versions that are in use if force is True
        for v in sorted(versions_in_use):
            self.metadata.hide_component_version(component, identifier, v)

    def _remove_single_version(
            self,
            component: str,
            identifier: str,
            version: int,
            recursive: bool = False,
            force: bool = False,
    ) -> None:
        """
        Remove a specific component version.

        Args:
            component: Component type
            identifier: Component identifier
            version: Version to remove
            recursive: Remove child components if True
            force: Skip confirmation if True
        """
        # Check if component exists
        component_info = None
        try:
            component_info = self.metadata.get_component(
                component, identifier, version=version
            )
        except NonExistentMetadataError:
            logging.warn(
                f'Component {component}:{identifier}:{version} has already been removed'
            )
            return

        component_uuid = component_info['uuid']

        # Check if component is used by others
        has_parents = self.metadata.component_version_has_parents(
            component, identifier, version
        )
        if has_parents:
            parents = self.metadata.get_component_version_parents(component_uuid)
            if not force:
                raise exceptions.ComponentInUseError(
                    f'{component_uuid} is involved in other components: {parents}'
                )
            logging.warn(f'Component {component_uuid} is in use: {parents}, but force=True')

        # Confirm deletion
        should_proceed = False
        if force:
            should_proceed = True
        else:
            should_proceed = click.confirm(
                f'You are about to delete {component}/{identifier}/{version}, '
                'are you sure?',
                default=False,
            )

        if not should_proceed:
            return

        # Load component
        component_obj = self.load(component, identifier, version=version)

        # Run component cleanup
        component_obj.cleanup()

        # Find and delete artifacts
        self._manage_artifacts(component_uuid, component_info, operation="delete")

        # Remove from metadata
        self.metadata.delete_component_version(component, identifier, version=version)
        self.metadata.delete_parent_child_relationships(component_uuid)

        # Handle recursive deletion
        if recursive:
            children = component_obj.get_children(deep=True)
            children = component_obj.sort_components(children)
            children.reverse()  # Reverse to handle deepest dependencies first

            for child in children:
                if child.version is None:
                    logging.warn(
                        f'Found uninitialized component {child.huuid}, skipping...'
                    )
                    continue

                try:
                    self._remove_single_version(
                        child.component,
                        child.identifier,
                        child.version,
                        recursive=False,
                        force=force,
                    )
                except exceptions.ComponentInUseError as e:
                    if force:
                        logging.warn(
                            f'Component {child.huuid} is in use: {e}\n'
                            'Skipping since force=True...'
                        )
                    else:
                        raise e

    def load_all(self, component: str, **kwargs) -> t.List[Component]:
        """
        Load all instances of a component type with optional filtering.

        Args:
            component: Component type to load
            **kwargs: Attribute filters to apply

        Returns:
            List of matching components
        """
        identifiers = self.metadata.show_components(component=component)
        result = []

        for identifier in identifiers:
            try:
                comp = self.load(component, identifier)

                # Filter by attributes if specified
                matches_all_filters = True
                for key, value in kwargs.items():
                    if not hasattr(comp, key) or getattr(comp, key) != value:
                        matches_all_filters = False
                        break

                if matches_all_filters:
                    result.append(comp)

            except NonExistentMetadataError:
                continue

        return result

    def load(
            self,
            component: str,
            identifier: t.Optional[str] = None,
            version: t.Optional[int] = None,
            uuid: t.Optional[str] = None,
            huuid: t.Optional[str] = None,
            allow_hidden: bool = False,
            overrides: t.Optional[t.Dict] = None,
    ) -> Component:
        """
        Load a component using various ways to identify it.

        Args:
            component: Component type to load
            identifier: Component identifier
            version: Specific version to load
            uuid: Component UUID
            huuid: Human-readable UUID
            allow_hidden: Include hidden components if True
            overrides: Dictionary of attribute overrides

        Returns:
            Loaded component instance
        """
        if overrides is None:
            overrides = {}

        info = None
        component_uuid = None

        # Load component info based on provided parameters
        if version is not None and identifier is not None:
            # Case 1: Load by version and identifier
            info = self.metadata.get_component(
                component=component,
                identifier=identifier,
                version=version,
                allow_hidden=allow_hidden,
            )
            component_uuid = info['uuid']

        elif huuid is not None:
            # Case 2: Load by human-readable UUID
            component_uuid = huuid.split(':')[-1]

        elif uuid is not None:
            # Case 3: Load by UUID
            component_uuid = uuid

        elif identifier is not None:
            # Case 4: Load by identifier (latest version)
            logging.info(f'Loading ({component}, {identifier}) from metadata...')
            info = self.metadata.get_component(
                component=component,
                identifier=identifier,
                allow_hidden=allow_hidden,
            )

        else:
            # Case 5: Missing required parameters
            raise ValueError("Must specify identifier, uuid, or huuid")

        # If we have UUID but not info, get the info
        if component_uuid is not None and info is None:
            info = self.metadata.get_component_by_uuid(
                component=component,
                uuid=component_uuid,
            )

        # Apply overrides
        for key, value in overrides.items():
            info[key] = value

        # Handle component loading based on info type
        if component_uuid is not None and identifier is None:
            # Clone builds to prevent cache modification
            builds = {}
            if '_builds' in info:
                for k, v in info['_builds'].items():
                    builds[k] = v.copy()
                    builds[k]['identifier'] = k.split(':')[-1]

            # Filter out _builds from info
            info_without_builds = {}
            for k, v in info.items():
                if k != '_builds':
                    info_without_builds[k] = v

            return BaseType().decode_data(
                info_without_builds,
                builds=builds,
                db=self,
            )
        else:
            builds = {}
            if '_builds' in info:
                builds = info['_builds']

            return ComponentType().decode_data(
                info,
                builds=builds,
                db=self,
            )

    def _manage_artifacts(
            self,
            uuid: str,
            info: t.Dict,
            operation: str
    ) -> t.Optional[t.Any]:
        """
        Manage artifacts associated with a component.

        Args:
            uuid: Component UUID
            info: Component information
            operation: Operation to perform ("save", "delete", "find")

        Returns:
            Operation result or None
        """
        # Find artifacts in the component info
        blobs = recursive_find(
            info, lambda v: isinstance(v, str) and v.startswith('&:blob:')
        )
        files = recursive_find(
            info, lambda v: isinstance(v, str) and v.startswith('&:file:')
        )

        # Extract artifact IDs
        artifact_ids = []
        for blob in blobs:
            artifact_ids.append(blob.split(":")[-1])
        for file in files:
            artifact_ids.append(file.split(":")[-1])

        artifacts = {'blobs': blobs, 'files': files}

        # Perform requested operation
        if operation == "save":
            return self.artifact_store.save_artifact(info)

        elif operation == "delete":
            for artifact_id in artifact_ids:
                relation_uuids = self.metadata.get_artifact_relations(
                    artifact_id=artifact_id
                )

                # Only delete if this is the only component using the artifact
                if len(relation_uuids) == 1 and relation_uuids[0] == uuid:
                    self.artifact_store.delete_artifact([artifact_id])
                    self.metadata.delete_artifact_relation(
                        uuid=uuid, artifact_ids=artifact_id
                    )
            return None

        elif operation == "find":
            return artifact_ids, artifacts

        else:
            raise ValueError(f"Unknown artifact operation: {operation}")

    def select_nearest(
            self,
            like: t.Union[t.Dict, Document],
            vector_index: str,
            ids: t.Optional[t.Sequence[str]] = None,
            outputs: t.Optional[Document] = None,
            n: int = 100,
    ) -> t.Tuple[t.List[str], t.List[float]]:
        """
        Perform a vector similarity search.

        Args:
            like: Query document or dictionary
            vector_index: Name of the vector index to search
            ids: Optional list of IDs to limit search scope
            outputs: Optional seed outputs document
            n: Number of results to return

        Returns:
            Tuple of (matching_ids, similarity_scores)
        """
        # Convert dictionary to Document if needed
        if not isinstance(like, Document):
            if not isinstance(like, dict):
                raise TypeError(f"Expected Document or dict, got {type(like)}")
            like = Document(like)

        # Load vector index
        logging.info('Getting vector index')
        vector_index = self.load('VectorIndex', vector_index)

        # Prepare outputs
        outs = {}
        if outputs is not None:
            outs = outputs.encode()
            if not isinstance(outs, dict):
                raise TypeError(f'Expected dict, got {type(outputs)}')

        logging.info(str(outs))
        return vector_index.get_nearest(like, ids=ids, n=n, outputs=outs)

    def disconnect(self) -> None:
        """Gracefully shutdown the Datalayer."""
        logging.info("Disconnecting from Cluster")
        self.cluster.disconnect()

    def get_cfg(self) -> Config:
        """Get the configuration object for the datalayer."""
        if self._cfg is None:
            return s.CFG
        return self._cfg

    def set_cfg(self, cfg: Config) -> None:
        """Set the configuration object for the datalayer."""
        if not isinstance(cfg, Config):
            raise AssertionError("cfg must be a Config instance")
        self._cfg = cfg

    # Property for config access
    cfg = property(get_cfg, set_cfg)

    def execute(self, query: str) -> t.Any:
        """
        Execute a native database query.

        Args:
            query: Raw query string in the native database language

        Returns:
            Query results
        """
        return self.databackend.execute_native(query)