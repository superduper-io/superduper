import dataclasses as dc
import datetime
import time
import traceback
import typing as t
import uuid
from traceback import format_exc

import click

from superduper import logging
from superduper.base import exceptions
from superduper.base.base import Base
from superduper.base.schema import Schema
from superduper.base.status import (
    JOB_PHASE_FAILED,
    JOB_PHASE_PENDING,
    JOB_PHASE_RUNNING,
    JOB_PHASE_SUCCESS,
    JOB_PHASE_UNINITIALIZED,
)
from superduper.components.cdc import CDC
from superduper.components.component import Component, init_status
from superduper.components.table import Table
from superduper.misc.importing import import_object
from superduper.misc.utils import merge_dicts

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer
    from superduper.components.component import Status


class Job(Base):
    """Job table.

    #noqa
    """

    queue: t.ClassVar[str] = '_apply'
    primary_id: t.ClassVar[str] = 'job_id'

    context: str
    component: str
    identifier: str
    uuid: str
    args: t.List = dc.field(default_factory=list)
    kwargs: t.Dict = dc.field(default_factory=dict)
    time: str = dc.field(default_factory=lambda: str(datetime.datetime.now()))
    job_id: t.Optional[str] = dc.field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    status: t.Dict = dc.field(default_factory=init_status)
    dependencies: t.List[str] = dc.field(default_factory=list)

    def get_status(self, db):
        """Get the status of the job.

        :param db: Datalayer instance
        """
        return db['Job'].get(job_id=self.job_id)['status']

    def wait(self, db: 'Datalayer', heartbeat: float = 1, timeout: int = 60):
        """Wait for the job to finish.

        :param db: Datalayer instance
        :param heartbeat: time to wait between checks
        :param timeout: timeout in seconds

        :raises Exception: if job remains uninitialized or pending for more than timeout seconds
        :raises Exception: if job fails
        :raises Exception: if job exceeds total timeout duration
        :return: final job status
        """
        start = time.time()
        while True:
            status = self.get_status(db)
            elapsed_time = time.time() - start

            # Check if timeout has been exceeded
            if elapsed_time > timeout:
                if (
                    status['phase'] == JOB_PHASE_UNINITIALIZED
                    or status['phase'] == JOB_PHASE_PENDING
                ):
                    # FIXME: should we even raise an error in this case, or just abort the job ?
                    err_msg = f'Job {self.job_id} remained in {status["phase"]} state for more than {timeout} seconds'
                    raise exceptions.TimeoutError(err_msg)
                else:
                    # FIXME: should we even raise an error in this case, or just abort the job ?
                    err_msg = f'Job {self.job_id} timed out after {timeout} seconds'
                    raise exceptions.TimeoutError(err_msg)

            if status['phase'] == JOB_PHASE_UNINITIALIZED:
                logging.info(f'Job {self.job_id} is uninitialized')
            elif status['phase'] == JOB_PHASE_PENDING:
                logging.info(f'Job {self.job_id} is pending')
            elif status['phase'] == JOB_PHASE_RUNNING:
                logging.info(f'Job {self.job_id} is running')
            else:
                break
            time.sleep(heartbeat)

        logging.info(f'Job {self.job_id} finished with status: {status}')
        if status['phase'] == JOB_PHASE_FAILED:
            # FIXME: should we even raise an error in this case, or just abort the job ?
            err_msg = f'Job {self.job_id} failed: {status}'
            raise exceptions.InternalError(err_msg)

        return status

    @property
    def huuid(self):
        """Return the hashed uuid."""
        return f'{self.component}:{self.identifier}:{self.uuid}.{self.method}'

    def get_args_kwargs(self, futures):
        """Get args and kwargs for job execution.

        :param futures: dict of futures
        """
        from superduper.backends.base.scheduler import Future

        dependencies = []
        if self.dependencies:
            dependencies = [futures[k] for k in self.dependencies if k in futures]
        args = []
        for arg in self.args:
            if isinstance(arg, Future):
                args.append(futures[arg.job_id])
            else:
                args.append(arg)
        kwargs = {}
        for k, v in self.kwargs.items():
            if isinstance(v, Future):
                kwargs[k] = futures[v.job_id]
            else:
                kwargs[k] = v
        kwargs['dependencies'] = dependencies
        return args, kwargs

    def run(self, db: 'Datalayer') -> None:
        """Run the job.

        :param db: Datalayer instance
        """
        try:
            logging.info(f'Running job {self.job_id}')
            db.metadata.set_job_status(
                self.job_id,
                {
                    "phase": JOB_PHASE_RUNNING,
                    "reason": "job started",
                },
            )
            logging.info(f'Loading component for job {self.job_id}')
            component = db.load(component=self.component, uuid=self.uuid)
            component.setup()
            logging.info(f'Executing method for job {self.job_id}')
            method = getattr(component, self.method)
            method(*self.args, **self.kwargs)
        except Exception as e:
            logging.error(
                f'Error running job {self.huuid}: {e}. Traceback: {traceback.format_exc()}'
            )
            db.metadata.set_job_status(
                self.job_id,
                {
                    "phase": JOB_PHASE_FAILED,
                    "reason": f"job failed: {str(e)}",
                    "message": format_exc(),
                },
            )
            # Since all the state management goes through the database, we don't need
            # to raise the exception here.
            return

        logging.info(f'Updating metadata for job {self.job_id}. Phase: Success')
        db.metadata.set_job_status(
            self.job_id,
            {
                "phase": JOB_PHASE_SUCCESS,
                "reason": "job succeeded",
            },
        )
        logging.success(f"Job {self.job_id} completed")

    def execute(
        self,
        db: 'Datalayer',
    ):
        """Execute the job event.

        :param db: Datalayer instance
        """
        meta = {k: v for k, v in self.dict().items() if k not in {'genus', 'queue'}}
        db.metadata.create_job(meta)
        return db.cluster.compute.submit(self)


class ParentChildAssociations(Base):
    """Parent-child associations table.

    :param parent_component: parent component type
    :param parent_identifier: parent component identifier
    :param parent_uuid: parent uuid
    :param child_component: child component type
    :param child_identifier: child component identifier
    :param child_uuid: child component uuid
    """

    parent_component: str
    parent_identifier: str
    parent_uuid: str
    child_component: str
    child_identifier: str
    child_uuid: str


class ArtifactRelations(Base):
    """Artifact relations table.

    :param relation_id: relation identifier
    :param component: component type
    :param identifier: identifier of component
    :param uuid: UUID of component version
    :param artifact_id: UUID of component version
    """

    primary_id: t.ClassVar[str] = 'relation'
    relation_id: str
    component: str
    identifier: str
    uuid: str
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

    :param db: Datalayer instance for saving components.
    :param parent_db: Parent Datalayer instance for saving primary data.
    """

    def __init__(self, db: 'Datalayer', parent_db: 'Datalayer'):
        self.db = db
        self.parent_db = parent_db
        self._schema_cache: t.Dict[str, Schema] = {}
        self.primary_ids = {
            "Table": "uuid",
            "ParentChildAssociations": "uuid",
            "ArtifactRelations": "relation_id",
            "Job": "job_id",
        }

    def __getitem__(self, item: str):
        return self.db[item]

    def init(self):
        """Initialize the metadata store."""
        self.db.databackend.create_table_and_schema(
            'Table',
            Table.class_schema,
            primary_id='uuid',
        )

        try:
            r = self.db['Table'].get(identifier='Table')
        except exceptions.NotFound:
            r = None

        if r is None:
            r = Table(
                identifier='Table',
                primary_id='uuid',
                is_component=True,
                path='superduper.components.table.Table',
            ).encode()
            r['version'] = 0

            self.db.databackend.insert('Table', [r])

        r = self.get_component('Table', 'Table')

        assert r is not None, 'Something went wrong in initializing the metadata store'

        self.create(ParentChildAssociations)
        self.create(ArtifactRelations)
        self.create(Job)

    def get_primary_id(self, table: str):
        """Get the primary id of a table.

        :param table: table name.
        """
        pid = self.primary_ids.get(table)

        if pid is None:
            pid = self.get_component(component="Table", identifier=table, version=0)[
                "primary_id"
            ]
            self.primary_ids[table] = pid

        return pid

    def create_table_and_schema(
        self,
        identifier: str,
        schema: 'Schema',
        primary_id: str,
        is_component: bool,
    ):
        """Create a table and schema in the metadata store.

        :param identifier: table name
        :param schema: schema of the table
        :param primary_id: primary id of the table
        :param is_component: whether the table is a component
        """
        if is_component:
            self.db.databackend.create_table_and_schema(
                identifier, schema, primary_id=primary_id
            )
        else:
            self.parent_db.databackend.create_table_and_schema(
                identifier, schema, primary_id=primary_id
            )

    def drop(self, force: bool = False):
        """Drop the metadata store.

        :param force: whether to force drop the metadata store.
        """
        if not force and not click.confirm(
            'Are you sure you want to drop the metadata store?'
        ):
            logging.warn('Aborting drop of metadata store')
        self.db.databackend.drop(force=force)
        self.db.artifact_store.drop(force=force)

    def is_component(self, table: str):
        """Check if a table is a component.

        :param table: table name.
        """
        return self.get_component('Table', table)

    def get_schema(self, table: str):
        """Get the schema of a table.

        :param table: table name.
        """
        if table in metaclasses:
            return metaclasses[table].class_schema

        if table in self._schema_cache:
            return self._schema_cache[table]

        r = self.get_component('Table', table)
        fields = r['fields']

        # TODO this seems to be to do with json_native
        if isinstance(fields, str):
            import json

            fields = json.loads(fields)
        schema = Schema.build(**fields)
        self._schema_cache[table] = schema
        return schema

    def create(self, cls: t.Type[Base]):
        """
        Create a table in the metadata store.

        :param cls: class to create
        """
        try:
            r = self.get_component('Table', cls.__name__)
            if r is not None:
                return
        except exceptions.NotFound:
            pass

        pid = self.db.databackend.id_field
        if issubclass(cls, Component):
            pid = 'uuid'
        elif getattr(cls, 'primary_id', None) is not None:
            pid = getattr(cls, 'primary_id', None)

        if issubclass(cls, Component) or cls.__name__ in metaclasses:
            self.db.databackend.create_table_and_schema(
                cls.__name__, cls.class_schema, primary_id=pid
            )
        else:
            self.parent_db.databackend.create_table_and_schema(
                cls.__name__, cls.class_schema, primary_id=pid
            )

        t = Table(
            identifier=cls.__name__,
            path=f'{cls.__module__}.{cls.__name__}',
            primary_id=pid,
            is_component=issubclass(cls, Component),
        )

        r = t.dict()
        try:
            r.pop('_path')
        except KeyError:
            pass
        r = {**r, 'version': 0, 'uuid': t.uuid}

        self.db['Table'].insert([r])

        return t

    def delete_parent_child_relationships(
        self, parent_component: str, parent_identifier: str
    ):
        """
        Delete parent-child mappings.

        :param parent_component: parent component type
        :param parent_identifier: parent component identifier
        """
        self.db['ParentChildAssociations'].delete(
            {
                'parent_component': parent_component,
                'parent_identifier': parent_identifier,
            }
        )

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
                raise exceptions.NotFound(component, path)

            assert r['path'] == path, msg
        except exceptions.NotFound:
            assert path is not None
            cls = import_object(path)
            self.create(cls)

        if '_path' in info:
            del info['_path']

        self.db[component].insert([info], raw=raw)

    def create_parent_child(
        self,
        parent_component: str,
        parent_identifier: str,
        parent_uuid: str,
        child_component: str,
        child_identifier: str,
        child_uuid: str,
    ):
        """
        Create a parent-child relationship between two components.

        :param parent_component: parent component type
        :param parent_identifier: parent component identifier
        :param parent_uuid: parent uuid
        :param child_component: child component type
        :param child_identifier: child component identifier
        :param child_uuid: child component uuid
        """
        r = {
            'parent_component': parent_component,
            'parent_identifier': parent_identifier,
            'parent_uuid': parent_uuid,
            'child_component': child_component,
            'child_identifier': child_identifier,
            'child_uuid': child_uuid,
        }

        self.db['ParentChildAssociations'].insert([r])

    def create_artifact_relation(self, component, identifier, uuid, artifact_ids):
        """
        Create a relation between an artifact and a component version.

        :param component: type of component
        :param identifier: identifier of component
        :param uuid: UUID of component version
        :param artifact_ids: artifact
        """
        artifact_ids = (
            [artifact_ids] if not isinstance(artifact_ids, list) else artifact_ids
        )
        data = []
        for artifact_id in artifact_ids:
            data.append(
                {
                    'component': component,
                    'identifier': identifier,
                    'uuid': uuid,
                    'artifact_id': artifact_id,
                }
            )

        if data:

            self.db['ArtifactRelations'].insert(data, raw=True)

    def delete_artifact_relation(
        self, component: str, identifier: str, artifact_ids: t.List[str]
    ):
        """
        Delete a relation between an artifact and a component version.

        :param component: type of component
        :param identifier: identifier of component
        :param artifact_ids: artifact ids
        """
        artifact_ids = (
            [artifact_ids] if not isinstance(artifact_ids, list) else artifact_ids
        )

        for artifact_id in artifact_ids:
            self.db['ArtifactRelations'].delete(
                {
                    'component': component,
                    'identifier': identifier,
                    'artifact_id': artifact_id,
                }
            )

    def get_artifact_relations_for_component(self, component, identifier):
        """
        Get all relations between an artifact and a component version.

        :param component: type of component
        :param identifier: identifier of component
        """
        t = self.db['ArtifactRelations']
        relations = t.filter(
            t['component'] == component, t['identifier'] == identifier
        ).execute()
        ids = [relation['artifact_id'] for relation in relations]
        return ids

    def get_artifact_relations_for_artifacts(self, artifact_ids: t.List[str]):
        """
        Get all relations between an artifact and a component version.

        :param artifact_ids: artifacts
        """
        t = self.db['ArtifactRelations']
        return t.filter(t['artifact_id'].isin(artifact_ids)).execute()

    def set_job_status(self, job_id: str, status_update: t.Dict):
        """
        Set the status of a job.

        :param job_id: job identifier
        :param status_update: status to set
        """
        job = self.db['Job'].get(job_id=job_id)
        if job is None:
            raise exceptions.NotFound("job", job_id)

        status = merge_dicts(job['status'], status_update)

        children = status.get('children', {})
        failed_children = [
            v for v in children.values() if v.get('phase') == JOB_PHASE_FAILED
        ]

        if failed_children:
            status['phase'] = JOB_PHASE_FAILED
            n_failed = len(failed_children)
            status['reason'] = (
                f"{n_failed} {'child' if n_failed == 1 else 'children'} failed"
            )

        self.db['Job'].update({'job_id': job_id}, 'status', status)

        if status['phase'] == JOB_PHASE_FAILED:
            key = f"Job/{job['component']}/{job['identifier']}/{job['uuid']}.{job['method']}"
            self.set_component_status(
                job['component'], job['uuid'], {'children': {key: status}}
            )

    def set_component_status(self, component: str, uuid: str, status_update: t.Dict):
        """
        Set the status of a component.

        :param component: type of component
        :param uuid: ``Component.uuid``
        :param status_update: Status document to set
        """
        r = self.db[component].get(uuid=uuid)
        status = merge_dicts(r['status'], status_update)

        if status['children'] and any(
            v['phase'] == JOB_PHASE_FAILED for v in status['children'].values()
        ):
            status['phase'] = JOB_PHASE_FAILED
            n_failed = len(
                [
                    v
                    for v in status['children'].values()
                    if v['phase'] == JOB_PHASE_FAILED
                ]
            )
            status['reason'] = '{} {} failed'.format(
                n_failed, 'child' if n_failed == 1 else 'children'
            )

        self.db[component].update({'uuid': uuid}, 'status', status)

        if status['phase'] == JOB_PHASE_FAILED:

            parents = self.get_component_version_parents(uuid)
            for parent in parents:
                self.set_component_status(
                    parent[0],
                    parent[1],
                    {
                        'phase': JOB_PHASE_FAILED,
                        'children': {
                            f'{component}/{r["identifier"]}/{uuid}': status,
                        },
                    },
                )

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
        return self.db['Job'].get(job_id=job_id)

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

    def show_jobs(self, component: str, identifier: str, status: str | None = None):
        """
        Show all jobs in the metadata store.

        :param component: type of component
        :param identifier: identifier of component
        :param status: status of job
        """
        filters = [
            self.db['Job']['component'] == component,
            self.db['Job']['identifier'] == identifier,
        ]

        if status is not None:
            filters.append(self.db['Job']['status'] == status)

        return self.db['Job'].filter(*filters).distinct('job_id')

    def show_components(self, component: str | None = None):
        """
        Show all components in the metadata store.

        :param component: type of component
        """
        if component is None:
            out = []

            t = self.db['Table']
            components = t.filter(t['is_component'] == True).distinct(  # noqa: E712
                'identifier'
            )

            for component in components:
                if component in metaclasses.keys():
                    continue

                identifiers = self.db[component].distinct('identifier')

                try:
                    out.extend(
                        [{'component': component, 'identifier': x} for x in identifiers]
                    )
                except ModuleNotFoundError as e:
                    logging.error(f'Component type not found: {component}; ', e)

            identifiers = self.db['Table'].distinct('identifier')

            out.extend([{'component': 'Table', 'identifier': x} for x in identifiers])
            return out

        return self.db[component].distinct('identifier')

    def show_cdc_tables(self):
        """List the tables used for CDC."""
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

    def delete_component(self, component: str, identifier: str):
        """
        Delete a component version from the metadata store.

        :param component: type of component
        :param identifier: identifier of component
        """
        self.db[component].delete({'identifier': identifier})

    def get_uuid(self, component: str, identifier: str, version: int):
        """
        Get the UUID of a component version.

        :param component: type of component
        :param identifier: identifier of component
        :param version: version of component
        """
        t = self.db[component]
        r = (
            t.filter(t['identifier'] == identifier, t['version'] == version)
            .select('uuid')
            .get()
        )
        if r is None:
            raise exceptions.NotFound(component, identifier)

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
            raise exceptions.NotFound(component, identifier)
        return max(versions)

    def get_component_by_uuid(self, component: str, uuid: str):
        """Get a component by UUID.

        :param component: type of component
        :param uuid: UUID of component
        """
        r = None
        if r is None:
            r = self.db[component].get(uuid=uuid, raw=True)

        if r is None:
            raise exceptions.NotFound(component, uuid)

        # TODO replace database query with cache query
        metadata = self.db['Table'].get(identifier=component)

        if metadata is None:
            raise exceptions.NotFound("Table", component)

        path = metadata['path']
        r['_path'] = path

        return r

    def get_latest_uuid(self, component: str, identifier: str):
        """Check if a component version has been updated.

        :param component: type of component
        :param identifier: identifier of component
        """
        return self.get_component(component, identifier)['uuid']

    def get_component(
        self,
        component: str,
        identifier: str,
        version: t.Optional[int] = None,
    ) -> t.Dict[str, t.Any]:
        """
        Get a component from the metadata store.

        :param component: type of component
        :param identifier: identifier of component
        :param version: version of component
        """
        if version is None:
            version = self.get_latest_version(
                component=component,
                identifier=identifier,
            )

        r = self.db[component].get(identifier=identifier, version=version, raw=True)

        if r is None:
            raise exceptions.NotFound(component, identifier)

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
        self.db[component].replace({'uuid': uuid}, info)

    def get_component_parents(self, component: str, identifier: str):
        """
        Get the parents of a component.

        :param component: type of component
        :param identifier: identifier of component
        """
        t = self.db['ParentChildAssociations']
        q = t.filter(
            t['child_component'] == component, t['child_identifier'] == identifier
        ).select('parent_component', 'parent_identifier')
        results = q.execute()
        return [(r['parent_component'], r['parent_identifier']) for r in results]

    def get_component_version_parents(self, uuid: str):
        """
        Get the parents of a component version.

        :param uuid: unique identifier of component version
        """
        t = self.db['ParentChildAssociations']
        q = t.filter(t['child_uuid'] == uuid).select('parent_component', 'parent_uuid')
        results = q.execute()
        return [(r['parent_component'], r['parent_uuid']) for r in results]
