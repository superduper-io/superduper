import dataclasses as dc
import datetime
import typing as t
import uuid

from superduper import logging
from superduper.backends.base.cache import Cache
from superduper.base.base import Base
from superduper.base.exceptions import DatabackendError
from superduper.base.schema import Schema
from superduper.components.cdc import CDC
from superduper.components.component import Component
from superduper.components.table import Table
from superduper.misc.importing import import_object


class NonExistentMetadataError(Exception):
    """Error to raise when a component type doesn't exist.

    # noqa
    """


class UniqueConstraintError(Exception):
    """Unique constraint error to raise when a component exists.

    # noqa
    """


# TODO merge with Event/Job
class Job(Base):
    """Job table.

    #noqa
    """

    context: str
    component: str
    identifier: str
    uuid: str
    args: t.List = dc.field(default_factory=list)
    kwargs: t.Dict = dc.field(default_factory=dict)
    time: str = dc.field(default_factory=lambda: str(datetime.datetime.now))
    job_id: t.Optional[str] = dc.field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    status: str = 'pending'
    dependencies: t.List[str] = dc.field(default_factory=list)


class ParentChildAssociations(Base):
    """Parent-child associations table.

    :param parent_component: parent component type
    :param parent_uuid: parent uuid
    :param child_component: child component type
    :param child_uuid: child component uuid
    """

    parent_component: str
    parent_uuid: str
    child_component: str
    child_uuid: str


class ArtifactRelations(Base):
    """Artifact relations table.

    :param component_id: UUID of component version
    :param artifact_id: UUID of component version
    """

    component_id: str
    artifact_id: str


metaclasses = {
    'Table': Table,
    'ParentChildAssociations': ParentChildAssociations,
    'ArtifactRelations': ArtifactRelations,
    'Job': Job,
}


class MetaDataStore:
    """
    Abstraction for storing meta-data separately from primary data.

    :param db: Datalayer instance.
    :param cache: Cache instance.
    """

    def __init__(self, db, cache: Cache | None = None):
        self.db = db
        self.cache = cache

        self.preset_components = {
            ('Table', 'Table'): Table(
                identifier='Table',
                primary_id='uuid',
                component=True,
                path='superduper.components.table.Table',
            ).encode(),
            ('Table', 'ParentChildAssociations'): Table(
                identifier='ParentChildAssociations',
                primary_id='uuid',
                component=True,
                path='superduper.base.metadata.ParentChildAssociations',
            ).encode(),
            ('Table', 'ArtifactRelations'): Table(
                identifier='ArtifactRelations',
                primary_id='uuid',
                component=True,
                path='superduper.base.metadata.ArtifactRelations',
            ).encode(),
            ('Table', 'Job'): Table(
                identifier='Job',
                primary_id='job_id',
                component=True,
                path='superduper.base.metadata.Job',
            ).encode(),
        }

        for v in self.preset_components.values():
            v['_path'] = 'superduper.components.table.Table'
            v['version'] = 0

        self.preset_uuids = {}
        for info in self.preset_components.values():
            self.preset_uuids[info['uuid']] = info

    def init(self):
        """Initialize the metadata store."""
        for cls in metaclasses.values():
            preset = self.preset_components.get(('Table', cls.__name__))
            self.db.databackend.create_table_and_schema(
                cls.__name__,
                cls.class_schema,
                primary_id=preset['primary_id'],
            )

    def get_schema(self, table: str):
        """Get the schema of a table.

        :param table: table name.
        """
        if table in metaclasses:
            return metaclasses[table].class_schema

        r = self.db['Table'].get(identifier=table)
        try:
            r = r.unpack()
            if r['path'] is not None:
                return import_object(r['path']).class_schema
            return Schema.build(**r['fields'])
        except AttributeError as e:
            if 'unpack' in str(e) and 'NoneType' in str(e):
                raise NonExistentMetadataError(f'{table} does not exist in metadata')
            raise e
        except TypeError as e:
            if 'NoneType' in str(e):
                raise NonExistentMetadataError(f'{table} does not exist in metadata')
            raise e

    def create(self, cls: t.Type[Base]):
        """
        Create a table in the metadata store.

        :param cls: class to create
        """
        try:
            r = self.db['Table'].get(identifier=cls.__name__)
        except DatabackendError as e:
            if 'not found' in str(e):
                self.db.databackend.create_table_and_schema(
                    'Table', Table.class_schema, 'uuid'
                )
                t = Table(
                    'Table',
                    path='superduper.components.table.Table',
                    primary_id='uuid',
                    component=True,
                )
                r = self.db['Table'].insert(
                    [{**t.dict(), 'version': 0, 'uuid': t.uuid}],
                )
            else:
                raise e

        if r:
            raise UniqueConstraintError(
                f'{cls.__name__} already exists in metadata with data: {r}'
            )

        pid = 'uuid' if issubclass(cls, Component) else self.db.databackend.id_field
        self.db.databackend.create_table_and_schema(
            cls.__name__, cls.class_schema, primary_id=pid
        )
        t = Table(
            identifier=cls.__name__,
            path=f'{cls.__module__}.{cls.__name__}',
            primary_id=pid,
            component=issubclass(cls, Component),
        )

        r = t.dict()
        try:
            r.pop('_path')
        except KeyError:
            pass
        r = {**r, 'version': 0, 'uuid': t.uuid}

        if self.cache:
            self.cache['Table', cls.__name__, r["uuid"]] = r

        self.db['Table'].insert([r])

        return t

    def delete_parent_child_relationships(self, parent_uuid: str):
        """
        Delete parent-child mappings.

        :param parent_uuid: parent component uuid
        """
        self.db['ParentChildAssociations'].delete({'parent_uuid': parent_uuid})

    def create_component(self, info: t.Dict, path: str, raw: bool = True):
        """
        Create a component in the metadata store.

        :param info: dictionary containing information about the component.
        :param path: path to the component class.
        :param raw: whether to insert raw data.
        """
        component = path.rsplit('.', 1)[1]

        try:
            msg = f'Component {component} with different path {path} already exists'
            r = self.get_component('Table', component)
            if r is None:
                raise NonExistentMetadataError
            assert r['path'] == path, msg
        except NonExistentMetadataError:
            assert path is not None
            cls = import_object(path)
            self.create(cls)

        if '_path' in info:
            del info['_path']

        if self.cache:
            self.cache[component, info["identifier"], info["uuid"]] = info

        self.db[component].insert([info], raw=raw)

    def create_parent_child(
        self,
        parent_component: str,
        parent_uuid: str,
        child_component: str,
        child_uuid: str,
    ):
        """
        Create a parent-child relationship between two components.

        :param parent_component: parent component type
        :param parent_uuid: parent uuid
        :param child_component: child component type
        :param child_uuid: child component uuid
        """
        self.db['ParentChildAssociations'].insert(
            [
                {
                    'parent_component': parent_component,
                    'parent_uuid': parent_uuid,
                    'child_component': child_component,
                    'child_uuid': child_uuid,
                }
            ]
        )

    def create_artifact_relation(self, uuid, artifact_ids):
        """
        Create a relation between an artifact and a component version.

        :param uuid: UUID of component version
        :param artifact_ids: artifact
        """
        artifact_ids = (
            [artifact_ids] if not isinstance(artifact_ids, list) else artifact_ids
        )
        data = []
        for artifact_id in artifact_ids:
            data.append({'component_id': uuid, 'artifact_id': artifact_id})

        if data:
            self.db['ArtifactRelations'].insert(data, raw=True)

    def delete_artifact_relation(self, uuid, artifact_ids):
        """
        Delete a relation between an artifact and a component version.

        :param uuid: UUID of component version
        :param artifact_ids: artifact ids
        """
        artifact_ids = (
            [artifact_ids] if not isinstance(artifact_ids, list) else artifact_ids
        )
        for artifact_id in artifact_ids:
            self.db['ArtifactRelations'].delete(
                {'component_id': uuid, 'artifact_id': artifact_id}
            )

    def get_artifact_relations(self, uuid=None, artifact_id=None):
        """
        Get all relations between an artifact and a component version.

        :param uuid: UUID of component version
        :param artifact_id: artifact
        """
        t = self.db['ArtifactRelations']
        if uuid is None and artifact_id is None:
            raise ValueError('Either `uuid` or `artifact_id` must be provided')
        elif uuid:
            relations = t.filter(t['component_id'] == uuid).execute()
            ids = [relation['artifact_id'] for relation in relations]
        else:
            relations = t.filter(t['artifact_id'] == artifact_id).execute()
            ids = [relation['component_id'] for relation in relations]
        return ids

    def set_component_status(self, component: str, uuid: str, status: str):
        """
        Set the status of a component.

        :param component: type of component
        :param uuid: ``Component.uuid``
        :param status: ``Status`` to set
        """
        self.db[component].update({'uuid': uuid}, key='status', value=status)

    def get_component_status(self, component: str, uuid: str):
        """
        Get the status of a component.

        :param component: type of component
        :param uuid: ``Component.uuid``
        """
        return self.db[component].get(uuid=uuid)['status']

    # ------------------ JOBS ------------------

    def get_job(self, job_id: str):
        """
        Get a job from the metadata store.

        :param job_id: job identifier
        """
        return self.db['Jobs'].get(job_id=job_id)

    def update_job(self, job_id: str, key: str, value: t.Any):
        """
        Update a job in the metadata store.

        :param job_id: job identifier
        :param key: key to be updated
        :param value: value to be updated
        """
        return self.db['Job'].update({'job_id': job_id}, key=key, value=value)

    def create_job(self, info: t.Dict):
        """Create a job in the metadata store.

        :param info: dictionary containing information about the job.
        """
        self.db['Job'].insert([info])

    def show_jobs(self, component: str, identifier: str):
        """
        Show all jobs in the metadata store.

        :param component: type of component
        :param identifier: identifier of component
        """
        return (
            self.db['Job']
            .filter(
                self.db['Job']['component'] == component,
                self.db['Job']['identifier'] == identifier,
            )
            .distinct('job_id')
        )

    def show_components(self, component: str | None = None):
        """
        Show all components in the metadata store.

        :param component: type of component
        """
        if component is None:
            out = []

            if self.cache:
                metadata = self.cache.get_with_component('Table')
                components = [r['identifier'] for r in metadata if r['component']]
            else:
                t = self.db['Table']
                components = t.filter(t['component'] == True).distinct(  # noqa: E712
                    'identifier'
                )

            for component in components:
                if component in metaclasses.keys():
                    continue

                if self.cache:
                    identifiers = sorted(
                        list(
                            set(
                                [
                                    r['identifier']
                                    for r in self.cache.get_with_component(component)
                                ]
                            )
                        )
                    )
                else:
                    identifiers = self.db[component].distinct('identifier')

                try:
                    out.extend(
                        [{'component': component, 'identifier': x} for x in identifiers]
                    )
                except ModuleNotFoundError as e:
                    logging.error(f'Component type not found: {component}; ', e)

            if self.cache:
                identifiers = sorted(
                    list(
                        set(
                            [
                                r['identifier']
                                for r in self.cache.get_with_component('Table')
                            ]
                        )
                    )
                )
            else:
                identifiers = self.db['Table'].distinct('identifier')

            out.extend([{'component': 'Table', 'identifier': x} for x in identifiers])
            return out

        if self.cache:
            return sorted(
                list(
                    set(
                        [
                            r['identifier']
                            for r in self.cache.get_with_component(component)
                        ]
                    )
                )
            )
        else:
            return self.db[component].distinct('identifier')

    def show_cdc_tables(self):
        """List the tables used for CDC."""
        from superduper.base.document import Document

        metadata = []

        if self.cache:
            metadata = self.cache.get_with_component('Table')
            metadata = [Document(r) for r in metadata]
        else:
            metadata = self.db['Table'].execute()

        cdc_classes = []
        for r in metadata:
            if r['path'] is None:
                continue
            cls = import_object(r['path'])
            r = r.unpack()
            if issubclass(cls, CDC):
                cdc_classes.append(r)

        cdc_tables = []
        for r in cdc_classes:
            cdc_tables.extend(self.db[r['identifier']].distinct('cdc_table'))
        return cdc_tables

    def show_cdcs(self, table):
        """
        Show the ``CDC`` components running on a given table.

        :param table: ``Table`` to consider.
        """
        cdc_classes = []

        if self.cache:
            metadata = self.cache.get_with_component('Table')
        else:
            metadata = self.db['Table'].execute()

        for r in metadata:
            if r['path'] is None:
                continue
            cls = import_object(r['path'])
            if issubclass(cls, CDC):
                cdc_classes.append(r)

        cdcs = []
        for r in cdc_classes:
            t = self.db[r['identifier']]
            results = t.filter(t['cdc_table'] == table).execute()
            for result in results:
                cdcs.extend([{'component': r['identifier'], 'uuid': result['uuid']}])
        return cdcs

    def show_component_versions(self, component: str, identifier: str):
        """
        Show all versions of a component in the metadata store.

        :param component: type of component
        :param identifier: identifier of component
        """
        t = self.db[component]
        return t.filter(t['identifier'] == identifier).distinct('version')

    def delete_component_version(self, component: str, identifier: str, version: int):
        """
        Delete a component version from the metadata store.

        :param component: type of component
        :param identifier: identifier of component
        :param version: version of component
        """
        if self.cache:
            try:
                uuid = self.get_uuid(component, identifier, version)
                del self.cache[component, identifier, uuid]
            except KeyError:
                pass
        self.db[component].delete({'identifier': identifier, 'version': version})

    def get_uuid(self, component: str, identifier: str, version: int):
        """
        Get the UUID of a component version.

        :param component: type of component
        :param identifier: identifier of component
        :param version: version of component
        """
        if self.cache:
            r = self.cache.get_with_component_identifier_version(
                component, identifier, version
            )
            if r is not None:
                assert isinstance(r, dict)
                return r['uuid']
        t = self.db[component]
        r = (
            t.filter(t['identifier'] == identifier, t['version'] == version)
            .select('uuid')
            .get()
        )
        if r is None:
            raise NonExistentMetadataError(
                f'{identifier} does not exist in metadata for {component}'
            )
        return r['uuid']

    def component_version_has_parents(
        self, component: str, identifier: str, version: int
    ):
        """
        Check if a component version has parents.

        :param component: type of component
        :param identifier: identifier of component
        :param version: version of component
        """
        uuid = self.get_uuid(component, identifier, version)
        return bool(self.get_component_version_parents(uuid))

    def get_latest_version(
        self, component: str, identifier: str, allow_hidden: bool = False
    ):
        """
        Get the latest version of a component.

        :param component: type of component
        :param identifier: identifier of component
        :param allow_hidden: whether to allow hidden components
        """
        t = self.db[component]
        versions = t.filter(t['identifier'] == identifier).distinct('version')

        if not versions:
            raise NonExistentMetadataError(
                f'{identifier} does not exist in metadata for {component}'
            )
        return max(versions)

    def get_component_by_uuid(self, component: str, uuid: str):
        """Get a component by UUID.

        :param component: type of component
        :param uuid: UUID of component
        """
        if uuid in self.preset_uuids:
            return self.preset_uuids[uuid]

        r = None
        if self.cache:
            try:
                r = self.cache.get_with_uuid(uuid)
            except KeyError:
                pass

        if r is None:
            r = self.db[component].get(uuid=uuid, raw=True)

        if r is None:
            raise NonExistentMetadataError(
                f'Object {uuid} does not exist in metadata for {component}'
            )

        if self.cache:
            self.cache[component, r['identifier'], r['uuid']] = r

        # TODO replace database query with cache query
        try:
            metadata = self.preset_components[('Table', component)]
        except KeyError:
            metadata = None
            if self.cache:
                metadata = self.cache.get_with_component_identifier('Table', component)
            if metadata is None:
                metadata = self.db['Table'].get(identifier=component)

            if metadata is None:
                raise NonExistentMetadataError(f'No such Table: {component}')

        path = metadata['path']
        r['_path'] = path

        return r

    def get_component(
        self,
        component: str,
        identifier: str,
        version: t.Optional[int] = None,
        allow_hidden: bool = False,  # TODO remove this
    ) -> t.Dict[str, t.Any]:
        """
        Get a component from the metadata store.

        :param component: type of component
        :param identifier: identifier of component
        :param version: version of component
        :param allow_hidden: whether to allow hidden components
        """
        if (component, identifier) in self.preset_components:
            return self.preset_components[(component, identifier)]

        r = None
        if self.cache:
            try:
                if version is None:
                    r = self.cache.get_with_component_identifier(component, identifier)
                else:
                    r = self.cache.get_with_component_identifier_version(
                        component, identifier, version
                    )
            except KeyError:
                pass

        if r is None:
            if version is None:
                version = self.get_latest_version(
                    component=component,
                    identifier=identifier,
                )
            r = self.db[component].get(identifier=identifier, version=version, raw=True)

        if r is None:
            raise NonExistentMetadataError(
                f'Object {identifier} does not exist in metadata for {component}'
            )

        if self.cache:
            self.cache[component, identifier, r["uuid"]] = r

        # TODO Does it have path or not? Let's pin it down please.
        if '_path' not in r:
            metadata = self.preset_components.get(('Table', component))
            if metadata is None and self.cache:
                metadata = self.cache.get_with_component_identifier('Table', component)
            if metadata is None:
                metadata = self.db['Table'].get(identifier=component)
            r['_path'] = metadata['path']

        return r

    def replace_object(self, component: str, uuid: str, info: t.Dict[str, t.Any]):
        """
        Replace an object in the metadata store.

        :param component: type of component.
        :param uuid: unique identifier of the object.
        :param info: dictionary containing information about the object.
        """
        if self.cache:
            self.cache[component, info["identifier"], uuid] = info
        self.db[component].replace({'uuid': uuid}, info)

    def remove_by_uuid(self, component: str, uuid: str):
        """
        Remove an object in the metadata store.

        :param component: type of component.
        :param uuid: unique identifier of the object.
        """
        if self.cache:
            self.cache.delete_uuid(uuid)

        self.db[component].delete({'uuid': uuid})

    def get_component_version_parents(self, uuid: str):
        """
        Get the parents of a component version.

        :param uuid: unique identifier of component version
        """
        t = self.db['ParentChildAssociations']
        q = t.filter(t['child_uuid'] == uuid).select('parent_component', 'parent_uuid')
        results = q.execute()
        return [(r['parent_component'], r['parent_uuid']) for r in results]
