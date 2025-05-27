import contextvars
import dataclasses as dc
import datetime
import time
import traceback
import typing as t
import uuid
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from traceback import format_exc

import click
import networkx as nx

from superduper import logging
from superduper.base import exceptions
from superduper.base.base import Base
from superduper.base.schema import Schema
from superduper.base.status import (
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_SUCCESS,
    STATUS_UNINITIALIZED,
)
from superduper.components.cdc import CDC
from superduper.components.component import Component, init_status
from superduper.components.table import Table
from superduper.misc.importing import import_object

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer
    from superduper.base.event import CreateTable


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
    job_id: str = dc.field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    details: t.Dict = dc.field(default_factory=lambda: init_status()[1])
    status: str = STATUS_UNINITIALIZED
    dependencies: t.List[str] = dc.field(default_factory=list)
    compute_kwargs: t.Dict = dc.field(default_factory=dict)
    result: t.List | None = None
    inverse_dependencies: t.List[str] = dc.field(default_factory=list)

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
            status = self.get_status(db)[0]
            elapsed_time = time.time() - start

            # Check if timeout has been exceeded
            if elapsed_time > timeout:
                if status == STATUS_UNINITIALIZED or status == STATUS_PENDING:
                    # FIXME: should we even raise an error in this case, or just abort the job ?
                    err_msg = f'Job {self.job_id} remained in {status} state for more than {timeout} seconds'
                    raise exceptions.TimeoutError(err_msg)
                else:
                    # FIXME: should we even raise an error in this case, or just abort the job ?
                    err_msg = f'Job {self.job_id} timed out after {timeout} seconds'
                    raise exceptions.TimeoutError(err_msg)

            if status == STATUS_UNINITIALIZED:
                logging.info(f'Job {self.job_id} is uninitialized')
            elif status == STATUS_PENDING:
                logging.info(f'Job {self.job_id} is pending')
            elif status == STATUS_RUNNING:
                logging.info(f'Job {self.job_id} is running')
            else:
                break
            time.sleep(heartbeat)

        logging.info(f'Job {self.job_id} finished with status: {status}')
        if status == STATUS_FAILED:
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

    @staticmethod
    def fix(db, context: str):
        """Fix the job.

        :param db: Datalayer instance
        :param context: context of the job
        """
        failed_jobs = (
            db['Job']
            .filter(
                db['Job']['context'] == context, db['Job']['status'] == STATUS_FAILED
            )
            .execute(decode=True)
        )
        lookup = {j.job_id: j for j in failed_jobs}

        G = nx.DiGraph()
        for j in failed_jobs:
            if j.dependencies:
                for dep in j.dependencies:
                    if dep in lookup:
                        G.add_edge(dep, j.job_id)

        input_nodes = [dep for dep in G.nodes() if G.in_degree(dep) == 0]

        for j in input_nodes:
            job = lookup[j]
            for k, v in job.kwargs.items():
                if isinstance(v, dict) and 'job_id' in v:
                    logging.info(f'Getting upstream result for {job.huuid}')
                    job.kwargs[k] = db['Job'].get(job_id=v['job_id'])['result']
                    logging.info(f'Getting upstream result for {job.huuid}... DONE')

        sorted_job_ids = list(nx.topological_sort(G))
        sorted_jobs = [lookup[j] for j in sorted_job_ids]

        db['Job'].update(
            {'context': context, 'status': STATUS_FAILED},
            'details',
            {'reason': 'job was resubmitted', 'message': 'job was resubmitted'},
        )
        db['Job'].update(
            {'context': context, 'status': STATUS_FAILED},
            'status',
            STATUS_PENDING,
        )

        from superduper.backends.base.scheduler import Future

        for j in sorted_jobs:
            for k, v in j.kwargs.items():
                if isinstance(v, dict) and 'job_id' in v:
                    j.kwargs[k] = Future(job_id=v['job_id'])

            args = list(j.args)
            for i, arg in enumerate(args):
                if isinstance(arg, dict) and 'job_id' in arg:
                    args[i] = Future(job_id=arg['job_id'])
            j.args = args

        for j in sorted_jobs:
            logging.info(f'Executing job {j.huuid}')
            j.execute(db)
            logging.info(f'Executing job {j.huuid}... DONE')

        db.cluster.compute.release_futures(context)
        return

    def set_failed(self, db: 'Datalayer', reason: str, message: str | None = None):
        """Set the job status to failed.

        :param db: Datalayer instance
        :param reason: reason for failure
        :param message: error message
        """
        logging.info(f'Setting job status {self.job_id} to failed')

        details = {
            'reason': reason,
            'message': message,
            'last_change_time': str(datetime.datetime.now()),
        }

        db['Job'].update({'job_id': self.job_id}, 'details', details)
        db['Job'].update({'job_id': self.job_id}, 'status', STATUS_FAILED)

        logging.info(
            f'Setting job status for inverse dependencies to failed: {self.inverse_dependencies}'
        )
        for idep in self.inverse_dependencies:
            logging.info(f'Setting downstream job {idep} status to failed')
            job = db['Job'].get(job_id=idep, decode=True)
            job.set_failed(
                db, reason=f"Upstream dependency {self.job_id} failed", message=None
            )
        logging.info(f'Setting job status {self.job_id} to failed... DONE')

        db.metadata.set_component_failed(
            component=self.component,
            uuid=self.uuid,
            context=self.context,
            failed_child=(
                f'Job/{self.huuid}',
                {
                    'reason': reason,
                    'message': message,
                    'context': self.context,
                },
            ),
        )

    def set_status(
        self,
        db: 'Datalayer',
        status: str,
        reason: str,
        message: str | None = None,
        was_broken: bool = False,
    ):
        """
        Set the status of a job.

        :param db: Datalayer instance
        :param status: status to set
        :param reason: reason for status change
        :param message: message for status change
        :param was_broken: whether the job was broken before
        """
        details: t.Dict = {
            'reason': reason,
            'message': message,
            'last_change_time': str(datetime.datetime.now()),
            'failed_children': {},
        }

        db['Job'].update({'job_id': self.job_id}, 'details', details)
        db['Job'].update({'job_id': self.job_id}, 'status', status)

        if was_broken and status == STATUS_SUCCESS:
            db.metadata.set_component_fixed(
                self.component,
                self.uuid,
                fixed_child=f'Job/{self.huuid}',
            )

    def get_status(self, db: 'Datalayer'):
        """Get the job status.

        :param db: Datalayer instance
        """
        j = db['Job'].get(job_id=self.job_id)
        if j is None:
            return STATUS_UNINITIALIZED
        return j['status'], j['details']

    def run(self, db: 'Datalayer'):
        """Run the job.

        :param db: Datalayer instance
        """
        was_broken = False
        try:
            status, details = self.get_status(db)
            if status == STATUS_FAILED:
                raise exceptions.InternalError(
                    'Upstream job failed',
                )
            elif (
                status == STATUS_PENDING and details['reason'] == 'job was resubmitted'
            ):
                was_broken = True
        except exceptions.NotFound:
            pass

        try:
            logging.info(f'Running job {self.huuid} {self.job_id}')
            self.set_status(
                db,
                STATUS_RUNNING,
                reason='job started',
            )
            logging.info(f'Loading component for job {self.job_id}')
            component = db.load(component=self.component, uuid=self.uuid)
            component.setup()
            logging.info(f'Executing method for job {self.job_id}')
            method = getattr(component, self.method)
            result = method(*self.args, **self.kwargs)
        except Exception as e:
            logging.error(
                f'Error running job {self.huuid}: {e}. Traceback: {traceback.format_exc()}'
            )
            self.set_failed(db=db, reason=str(e), message=format_exc())
            raise e

        logging.info(f'Updating metadata for job {self.job_id}. Phase: Success')
        self.set_status(
            db,
            STATUS_SUCCESS,
            reason='job succeeded',
            was_broken=was_broken,
        )
        if result:
            self.save_output(db, result)
        logging.success(f"Job {self.job_id} completed")
        return result

    def save_output(self, db: 'Datalayer', result: t.Any):
        """Save the output of the job.

        :param db: Datalayer instance
        :param result: result of the job
        """
        logging.info(f'Saving output for job {self.huuid}')
        db['Job'].update({'job_id': self.job_id}, 'result', result)
        logging.info(f'Saving output for job {self.huuid}... DONE')

    def execute(
        self,
        db: 'Datalayer',
    ):
        """Execute the job event.

        :param db: Datalayer instance
        """
        entry = db['Job'].get(job_id=self.job_id)
        if entry is None:
            db.metadata.create_job(dict(self.dict()))
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

_component_cache = contextvars.ContextVar('component_cache', default=None)


class MetaDataStore:
    """
    Abstraction for storing meta-data separately from primary data.

    :param db: Datalayer instance for saving components.
    :param parent_db: Parent Datalayer instance for saving primary data.
    """

    def __init__(self, db: 'Datalayer', parent_db: 'Datalayer'):
        self.db = db
        self.parent_db = parent_db
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

    @contextmanager
    def cache(self):
        """Context manager to enable and manage component class information caching."""
        current_cache = _component_cache.get()

        if current_cache is None:
            token = _component_cache.set({})
        else:
            token = None

        try:
            yield
        finally:
            if token is not None:
                _component_cache.reset(token)

    def _get_component_class_info(self, component):
        cache = _component_cache.get()

        if cache is not None and component in cache:
            return cache[component].copy()

        info = self.get_component('Table', component, version=0)

        if cache is not None and info:
            cache[component] = info.copy()

        return info

    def check_table_in_metadata(self, table: str):
        """Check if a table exists in the metadata store.

        :param table: table name.
        """
        if table in metaclasses:
            return True

        return table in self.db.databackend.list_tables()

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

    def create_tables_and_schemas(self, events: t.List['CreateTable']):
        """Create a table and schema in the metadata store.

        :param events: list of create table events.
        """
        metadata_tables = [e for e in events if e.is_component]
        main_tables = [e for e in events if not e.is_component]
        self.db.databackend.create_tables_and_schemas(metadata_tables)
        self.parent_db.databackend.create_tables_and_schemas(main_tables)

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
        return self._get_component_class_info(table).get('is_component', False)

    def get_schema(self, table: str):
        """Get the schema of a table.

        :param table: table name.
        """
        if table in metaclasses:
            return metaclasses[table].class_schema

        r = self._get_component_class_info(table)
        fields = r['fields']

        # TODO this seems to be to do with json_native
        if isinstance(fields, str):
            import json

            fields = json.loads(fields)
        schema = Schema.build(**fields)
        return schema

    def create(self, cls: t.Type[Base]):
        """
        Create a table in the metadata store.

        :param cls: class to create
        """
        try:
            r = self._get_component_class_info(cls.__name__)
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
            r = self._get_component_class_info(component)
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

    def set_component_failed(
        self,
        component: str,
        uuid: str,
        failed_child: t.Tuple[str, t.Dict] | None = None,
        message: str | None = None,
        reason: str | None = None,
        context: str | None = None,
    ):
        """
        Set the status of a component to failed.

        :param component: type of component
        :param uuid: ``Component.uuid``
        :param failed_child: child to fail
        :param message: error message
        :param reason: reason for failure
        :param context: context of the failure
        """
        r = self.db[component].get(uuid=uuid)
        details = r['details']

        if failed_child and failed_child[0] not in details['failed_children']:
            details['failed_children'][failed_child[0]] = failed_child[1]
        elif failed_child:
            return

        details['reason'] = (
            reason
            or f'{len(details["failed_children"])} {"children" if len(details["failed_children"]) > 1 else "child"} failed'
        )
        details['message'] = message
        details['last_change_time'] = str(datetime.datetime.now())
        details['context'] = context

        self.db[component].update({'uuid': uuid}, 'details', details)
        self.db[component].update({'uuid': uuid}, 'status', STATUS_FAILED)

        parents = self.get_component_version_parents(uuid)
        for parent in parents:
            self.set_component_failed(
                parent[0],
                parent[1],
                context=context,
                failed_child=(
                    f'{component}/{r["identifier"]}/{uuid}',
                    {
                        'reason': details['reason'],
                        'message': details['message'],
                    },
                ),
            )

    def set_component_fixed(
        self,
        component: str,
        uuid: str,
        fixed_child: str,
    ):
        """
        Set the status of a component to fixed.

        :param component: type of component
        :param uuid: ``Component.uuid``
        :param fixed_child: Child to fix
        """
        r = self.db[component].get(uuid=uuid)
        details = r['details']
        if fixed_child in details['failed_children']:
            details['failed_children'].pop(fixed_child)
        if not details['failed_children']:
            details['reason'] = 'The component is ready to use'
            details['message'] = None
            self.db[component].update({'uuid': uuid}, 'status', STATUS_RUNNING)
            parents = self.get_component_version_parents(uuid)
            for parent in parents:
                self.set_component_fixed(
                    parent[0],
                    parent[1],
                    f'{component}/{r["identifier"]}/{uuid}',
                )
        else:
            details['reason'] = f'{len(details["failed_children"])} children failed'
            details['message'] = None
        self.db[component].update({'uuid': uuid}, 'details', details)

    def set_component_status(
        self,
        component: str,
        uuid: str,
        status: str,
        reason: str,
        message: str | None = None,
    ):
        """
        Set the status of a component.

        :param component: type of component
        :param uuid: ``Component.uuid``
        :param status: Status to set
        :param reason: reason for status change
        :param message: message for status change
        """
        if status == STATUS_FAILED:
            raise exceptions.InternalError(
                'Use set_component_failed instead of set_component_status'
            )

        assert status in [
            STATUS_PENDING,
            STATUS_RUNNING,
            STATUS_SUCCESS,
        ], f'Invalid status: {status}'

        self.db[component].update(
            {'uuid': uuid},
            'details',
            {
                'reason': reason,
                'message': message,
                'last_change_time': str(datetime.datetime.now()),
                'failed_children': {},
            },
        )
        self.db[component].update({'uuid': uuid}, 'status', status)

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
                t = self.db[component]
                q = t.select('identifier', 'status', 'version')
                results = sorted(q.execute(), key=lambda x: x['version'])
                output = {}
                for r in results:
                    output[r['identifier']] = r['status']

                results = [
                    {'component': component, 'identifier': k, 'status': v}
                    for k, v in output.items()
                ]

                out.extend(results)

            identifiers = self.db['Table'].distinct('identifier')

            out.extend(
                [
                    {'component': 'Table', 'identifier': x, 'status': STATUS_RUNNING}
                    for x in identifiers
                ]
            )
            return out

        return self.db[component].distinct('identifier')

    def show_status(self, component: str, identifier: str | None = None):
        """
        Show the status of a component.

        :param component: type of component
        :param identifier: identifier of component
        """
        if component is None:
            raise exceptions.NotFound(component, identifier)
        t = self.db[component]
        if identifier:
            t = t.filter(t['identifier'] == identifier)
        t = t.select('status', 'version', 'identifier')
        results = sorted(t.execute(), key=lambda x: x['version'])
        output = {}
        for r in results:
            output[r['identifier']] = dict(r)
        output = list(output.values())
        for r in output:
            del r['version']
        return output

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

    def get_latest_versions(self):
        """Get the latest versions of a component."""
        components = self.show_components()
        out = []
        for component in components:
            t = self.db[component['component']]
            q = t.filter(t['identifier'] == component['identifier']).select(
                'version', 'uuid'
            )
            versions = q.execute()
            out.append({**component, **max(versions, key=lambda x: x['version'])})
        return out

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
        metadata = self._get_component_class_info(component)

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

        metadata = self.db['Table'].get(identifier=component, raw=True)
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
