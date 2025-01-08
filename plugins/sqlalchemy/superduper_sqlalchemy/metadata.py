import copy
import json
import threading
import typing as t
from collections import defaultdict
from contextlib import contextmanager

import click
from sqlalchemy import (
    Column,
    MetaData,
    Table,
    and_,
    create_engine,
    delete,
    insert,
    select,
)
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import sessionmaker
from superduper import logging
from superduper.backends.base.metadata import MetaDataStore, NonExistentMetadataError
from superduper.components.component import Status
from superduper.misc.colors import Colors

from superduper_sqlalchemy.db_helper import get_db_config


def _connect_snowflake():
    # In the Snowflake native apps framework, the
    # inbuild database is provided by env variables
    # and authentication is via OAuth with a
    # mounted token. In this case, as a convention
    # we connect with `"snowflake://"`

    import snowflake.connector

    def creator():
        import os

        return snowflake.connector.connect(
            host=os.environ['SNOWFLAKE_HOST'],
            port=int(os.environ['SNOWFLAKE_PORT']),
            account=os.environ['SNOWFLAKE_ACCOUNT'],
            authenticator='oauth',
            token=open('/snowflake/session/token').read(),
            warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
            database=os.environ['SNOWFLAKE_DATABASE'],
            schema=os.environ['SUPERDUPER_DATA_SCHEMA'],
        )

    return create_engine("snowflake://not@used/db", creator=creator)


class _Cache:
    def __init__(self):
        self._uuid2metadata: t.Dict[str, t.Dict] = {}
        self._type_id_identifier2metadata = defaultdict(dict)

    def replace_metadata(
        self, metadata, uuid=None, type_id=None, version=None, identifier=None
    ):
        metadata = copy.deepcopy(metadata)
        if 'dict' in metadata:
            dict_ = metadata['dict']
            del metadata['dict']
            metadata = {**metadata, **dict_}
        if uuid:
            self._uuid2metadata[uuid] = metadata

        version = metadata['version']
        type_id = metadata['type_id']
        identifier = metadata['identifier']
        self._type_id_identifier2metadata[(type_id, identifier)][version] = metadata
        return metadata

    def expire(self, uuid):
        if uuid in self._uuid2metadata:
            metadata = self._uuid2metadata[uuid]
            del self._uuid2metadata[uuid]
            type_id = metadata['type_id']
            identifier = metadata['identifier']
            if (type_id, identifier) in self._type_id_identifier2metadata:
                del self._type_id_identifier2metadata[(type_id, identifier)]

    def expire_identifier(self, type_id, identifier):
        if (type_id, identifier) in self._type_id_identifier2metadata:
            del self._type_id_identifier2metadata[(type_id, identifier)]

    def add_metadata(self, metadata):
        metadata = copy.deepcopy(metadata)
        if 'dict' in metadata:
            dict_ = metadata['dict']
            del metadata['dict']
            metadata = {**metadata, **dict_}

        self._uuid2metadata[metadata['uuid']] = metadata

        version = metadata['version']
        type_id = metadata['type_id']
        identifier = metadata['identifier']
        self._type_id_identifier2metadata[(type_id, identifier)][version] = metadata
        return metadata

    def get_metadata_by_uuid(self, uuid):
        return self._uuid2metadata.get(uuid)

    def get_metadata_by_identifier(self, type_id, identifier, version):
        metadata = self._type_id_identifier2metadata[(type_id, identifier)]
        if not metadata:
            return None
        if version is None:
            version = max(metadata.keys())
        return metadata.get(version)

    def update_metadata(self, metadata):
        self.add_metadata(metadata)


class SQLAlchemyMetadata(MetaDataStore):
    """
    Abstraction for storing meta-data separately from primary data.

    :param uri: URI to the databackend database.
    :param flavour: Flavour of the databackend.
    :param callback: Optional callback to create connection.
    """

    def __init__(
        self,
        uri: t.Optional[str] = None,
        flavour: t.Optional[str] = None,
        callback: t.Optional[t.Callable] = None,
    ):
        super().__init__(uri=uri, flavour=flavour)

        if callback:
            self.connection_callback = callback
        elif uri == 'snowflake://':
            name = 'snowflake'
            self.connection_callback = lambda: (_connect_snowflake(), name)
        else:
            assert isinstance(uri, str)
            name = uri.split('//')[0]
            self.connection_callback = lambda: (create_engine(uri), name)

        sql_conn, name = self.connection_callback()

        self.name = name
        self.conn = sql_conn
        self.dialect = sql_conn.dialect.name
        self._init_tables()

        self._lock = threading.Lock()
        self._cache = _Cache()
        self._init_cache()
        self._insert_flush = {
            'parent_child': [],
            'component': [],
            '_artifact_relations': [],
            'job': [],
        }
        self._parent_relation_cache = []
        self._batched = True

    def expire(self, uuid):
        """Expire metadata cache."""
        self._cache.expire(uuid)

    @property
    def batched(self):
        """Batched metadata updates."""
        return self._batched

    def _init_cache(self):
        with self.session_context() as session:
            stmt = select(self.component_table)
            res = self.query_results(self.component_table, stmt, session)
            for r in res:
                self._cache.add_metadata(r)

    def _get_db_table(self, table):
        if table == 'component':
            return self.component_table
        elif table == 'parent_child':
            return self.parent_child_association_table
        elif table == 'job':
            return self.job_table
        else:
            return self._table_mapping[table]

    def commit(self):
        """Commit execute."""
        if self._insert_flush:
            for table, flush in self._insert_flush.items():
                if flush:
                    with self.session_context() as session:
                        session.execute(insert(self._get_db_table(table)), flush)
                        self._insert_flush[table] = []
        with self.session_context() as session:
            session.commit()
        self._batched = False

    def reconnect(self):
        """Reconnect to sqlalchmey metadatastore."""
        sql_conn, name = self.connection_callback()
        self.conn = sql_conn

        # TODO: is it required to init after
        # a reconnect.
        self._init_tables()

    def _init_tables(self):
        # Get the DB config for the given dialect
        DBConfig = get_db_config(self.dialect)

        type_string = DBConfig.type_string
        type_string_long = DBConfig.type_string_long
        type_json_as_string = DBConfig.type_json_as_string
        type_json_as_text = DBConfig.type_json_as_text
        type_integer = DBConfig.type_integer
        type_boolean = DBConfig.type_boolean

        job_table_args = DBConfig.job_table_args
        parent_child_association_table_args = (
            DBConfig.parent_child_association_table_args
        )
        component_table_args = DBConfig.component_table_args

        metadata = MetaData()

        self.job_table = Table(
            'JOB',
            metadata,
            Column('context', type_string),
            Column('type_id', type_string),
            Column('identifier', type_string),
            Column('uuid', type_string),
            Column('args', type_json_as_string),
            Column('kwargs', type_json_as_text),
            Column('time', type_string),
            Column('job_id', type_string, primary_key=True),
            Column('method', type_string),
            Column('genus', type_string),
            Column('queue', type_string),
            Column('status', type_string),
            Column('dependencies', type_string_long),
            *job_table_args,
        )

        self.parent_child_association_table = Table(
            'PARENT_CHILD_ASSOCIATION',
            metadata,
            Column('parent_id', type_string, primary_key=True),
            Column('child_id', type_string, primary_key=True),
            *parent_child_association_table_args,
        )

        self.component_table = Table(
            "COMPONENT",
            metadata,
            # TODO rename with id -> uuid
            Column('id', type_string, primary_key=True),
            Column('identifier', type_string),
            Column('uuid', type_string),
            Column('version', type_integer),
            Column('hidden', type_boolean),
            Column('status', type_string),
            Column('type_id', type_string),
            Column('_path', type_string),
            Column('dict', type_json_as_text),
            Column('cdc_table', type_string),
            *component_table_args,
        )

        self.artifact_table = Table(
            'ARTIFACT_RELATIONS',
            metadata,
            Column('uuid', type_string),
            Column('artifact_id', type_string),
            *component_table_args,
        )

        self._table_mapping = {
            '_artifact_relations': self.artifact_table,
        }

        try:
            metadata.create_all(self.conn)
        except Exception as e:
            logging.error(f'Error creating tables: {e}')

    def _create_data(self, table_name, datas):
        table = self._table_mapping[table_name]
        with self.session_context(commit=not self.batched) as session:
            if not self.batched:
                for data in datas:
                    stmt = insert(table).values(**data)
                    session.execute(stmt)
            else:
                if table_name not in self._insert_flush:
                    self._insert_flush[table_name] = datas
                else:
                    self._insert_flush[table_name] += datas

    def _delete_data(self, table_name, filter):
        table = self._table_mapping[table_name]

        with self.session_context() as session:
            conditions = [getattr(table.c, k) == v for k, v in filter.items()]
            stmt = delete(table).where(*conditions)
            session.execute(stmt)

    def _get_data(self, table_name, filter):
        table = self._table_mapping[table_name]

        with self.session_context() as session:
            conditions = [getattr(table.c, k) == v for k, v in filter.items()]
            stmt = select(table).where(*conditions)
            res = self.query_results(table, stmt, session)
            return res

    def url(self):
        """Return the URL of the metadata store."""
        return self.conn.url + self.name

    def drop(self, force: bool = False):
        """Drop the metadata store.

        :param force: whether to force the drop (without confirmation)
        """
        if not force:
            if not click.confirm(
                f'{Colors.RED}[!!!WARNING USE WITH CAUTION AS YOU '
                f'WILL LOSE ALL DATA!!!]{Colors.RESET} '
                'Are you sure you want to drop all meta-data? ',
                default=False,
            ):
                logging.warn('Aborting...')

        try:
            self.job_table.drop(self.conn)
        except ProgrammingError as e:
            logging.warn(f'Error dropping job table: {e}')

        try:
            self.parent_child_association_table.drop(self.conn)
        except ProgrammingError as e:
            logging.warn(f'Error dropping parent-child association table: {e}')

        try:
            self.component_table.drop(self.conn)
        except ProgrammingError as e:
            logging.warn(f'Error dropping component table {e}')

        try:
            self.artifact_table.drop(self.conn)
        except ProgrammingError as e:
            logging.warn(f'Error dropping artifact table {e}')

    @contextmanager
    def session_context(self, commit=True):
        """Provide a transactional scope around a series of operations."""
        sm = sessionmaker(bind=self.conn)
        session = sm()
        try:
            yield session
            if commit:
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # --------------- COMPONENTS -----------------

    def _get_component_uuid(self, type_id: str, identifier: str, version: int):
        with self.session_context() as session:
            stmt = (
                select(self.component_table)
                .where(
                    self.component_table.c.type_id == type_id,
                    self.component_table.c.identifier == identifier,
                    self.component_table.c.version == version,
                )
                .limit(1)
            )
            res = self.query_results(self.component_table, stmt, session)
            return res[0]['id'] if res else None

    def _get_all_component_info(self):
        with self.session_context() as session:
            res = self.query_results(
                self.component_table,
                select(self.component_table),
                session=session,
            )
        return list(res)

    def component_version_has_parents(
        self, type_id: str, identifier: str, version: int
    ):
        """Check if a component version has parents.

        :param type_id: the type of the component
        :param identifier: the identifier of the component
        :param version: the version of the component
        """
        uuid = self._get_component_uuid(type_id, identifier, version)
        with self.session_context() as session:
            stmt = (
                select(self.parent_child_association_table)
                .where(
                    self.parent_child_association_table.c.child_id == uuid,
                )
                .limit(1)
            )
            res = self.query_results(self.parent_child_association_table, stmt, session)
            return len(res) > 0

    def create_component(
        self,
        info: t.Dict,
    ):
        """Create a component in the metadata store.

        :param info: the information to create the component
        """
        new_info = self._refactor_component_info(info)
        with self.session_context(commit=not self.batched) as session:
            if not self.batched:
                stmt = insert(self.component_table).values(new_info)
                session.execute(stmt)
            else:
                self._insert_flush['component'].append(copy.deepcopy(new_info))

            self._cache.add_metadata(new_info)

    def delete_parent_child(self, parent_id: str, child_id: str | None = None):
        """
        Delete parent-child relationships between two components.

        :param parent: parent component uuid
        :param child: child component uuid
        """
        if child_id:
            with self.session_context() as session:
                stmt = delete(self.parent_child_association_table).where(
                    self.parent_child_association_table.c.parent_id == parent_id,
                    self.parent_child_association_table.c.child_id == child_id,
                )
                session.execute(stmt)
            return

        with self.session_context() as session:
            stmt = delete(self.parent_child_association_table).where(
                self.parent_child_association_table.c.parent_id == parent_id,
            )
            session.execute(stmt)

    def create_parent_child(
        self,
        parent_id: str,
        child_id: str,
    ):
        """Create a parent-child relationship between two components.

        :param parent_id: the parent component
        :param child_id: the child component
        """
        import sqlalchemy

        try:
            self._parent_relation_cache.append((parent_id, child_id))
            with self.session_context(commit=not self.batched) as session:
                if not self.batched:
                    stmt = insert(self.parent_child_association_table).values(
                        parent_id=parent_id, child_id=child_id
                    )
                    session.execute(stmt)
                else:
                    if (parent_id, child_id) not in self._parent_relation_cache:
                        self._insert_flush['parent_child'].append(
                            {'parent_id': parent_id, 'child_id': child_id}
                        )

        except sqlalchemy.exc.IntegrityError:
            logging.warn(f'Skipping {parent_id} {child_id} since they already exists')

    def delete_component_version(self, type_id: str, identifier: str, version: int):
        """Delete a component from the metadata store.

        :param type_id: the type of the component
        :param identifier: the identifier of the component
        :param version: the version of the component
        """
        with self.session_context() as session:
            stmt = (
                self.component_table.select()
                .where(
                    self.component_table.c.type_id == type_id,
                    self.component_table.c.identifier == identifier,
                    self.component_table.c.version == version,
                )
                .limit(1)
            )
            res = self.query_results(self.component_table, stmt, session)
            cv = res[0] if res else None
            if cv:
                stmt_delete = delete(self.component_table).where(
                    self.component_table.c.id == cv['id']
                )
                session.execute(stmt_delete)
                self._cache.expire_identifier(type_id, identifier)

        if cv:
            self.delete_parent_child(cv['id'])

    def get_component_by_uuid(self, uuid: str, allow_hidden: bool = False):
        """Get a component by UUID.

        :param uuid: UUID of component
        :param allow_hidden: whether to load hidden components
        """
        if res := self._cache.get_metadata_by_uuid(uuid):
            return res
        with self.session_context() as session:
            stmt = (
                select(self.component_table)
                .where(
                    self.component_table.c.id == uuid,
                )
                .limit(1)
            )
            if not allow_hidden:
                stmt = stmt.where(self.component_table.c.hidden == allow_hidden)
            res = self.query_results(self.component_table, stmt, session)
            try:
                res = res[0]
                res = self._cache.add_metadata(res)
                return res
            except IndexError:
                raise NonExistentMetadataError(
                    f'Table with uuid: {uuid} does not exist'
                )

    def _get_component(
        self,
        type_id: str,
        identifier: str,
        version: int,
        allow_hidden: bool = False,
    ):
        """Get a component from the metadata store.

        :param type_id: the type of the component
        :param identifier: the identifier of the component
        :param version: the version of the component
        :param allow_hidden: whether to allow hidden components
        """
        if res := self._cache.get_metadata_by_identifier(type_id, identifier, version):
            return res
        with self.session_context() as session:
            stmt = select(self.component_table).where(
                self.component_table.c.type_id == type_id,
                self.component_table.c.identifier == identifier,
                self.component_table.c.version == version,
            )
            if not allow_hidden:
                stmt = stmt.where(self.component_table.c.hidden == allow_hidden)

            res = self.query_results(self.component_table, stmt, session)
            if res:
                res = res[0]
                res = self._cache.add_metadata(res)
                return res

    def get_component_version_parents(self, uuid: str):
        """Get the parents of a component version.

        :param uuid: the unique identifier of the component version
        """
        with self.session_context() as session:
            stmt = select(self.parent_child_association_table).where(
                self.parent_child_association_table.c.child_id == uuid,
            )
            res = self.query_results(self.parent_child_association_table, stmt, session)
            parents = [r['parent_id'] for r in res]
            return parents

    @classmethod
    def _refactor_component_info(cls, info):
        if 'hidden' not in info:
            info['hidden'] = False
        component_fields = [
            'identifier',
            'version',
            'hidden',
            'type_id',
            '_path',
            'cdc_table',
        ]
        new_info = {k: info.get(k) for k in component_fields}
        new_info['dict'] = {k: info[k] for k in info if k not in component_fields}
        new_info['id'] = new_info['dict']['uuid']
        new_info['uuid'] = new_info['dict']['uuid']
        return new_info

    def get_latest_version(
        self, type_id: str, identifier: str, allow_hidden: bool = False
    ):
        """Get the latest version of a component.

        :param type_id: the type of the component
        :param identifier: the identifier of the component
        :param allow_hidden: whether to allow hidden components
        """
        if res := self._cache.get_metadata_by_identifier(type_id, identifier, None):
            return res['version']
        with self.session_context() as session:
            stmt = (
                select(self.component_table)
                .where(
                    self.component_table.c.type_id == type_id,
                    self.component_table.c.identifier == identifier,
                    self.component_table.c.hidden == allow_hidden,
                )
                .order_by(self.component_table.c.version.desc())
                .limit(1)
            )
            res = self.query_results(self.component_table, stmt, session)
            versions = [r['version'] for r in res]
            if len(versions) == 0:
                raise FileNotFoundError(
                    f'Can\'t find {type_id}: {identifier} in metadata'
                )
            return versions[0]

    def hide_component_version(self, type_id: str, identifier: str, version: int):
        """Hide a component in the metadata store.

        :param type_id: the type of the component
        :param identifier: the identifier of the component
        :param version: the version of the component
        """
        with self.session_context() as session:
            stmt = (
                self.component_table.update()
                .where(
                    self.component_table.c.type_id == type_id,
                    self.component_table.c.identifier == identifier,
                    self.component_table.c.version == version,
                )
                .values(hidden=True)
            )
            session.execute(stmt)

    def _replace_object(
        self,
        info,
        identifier: str | None = None,
        type_id: str | None = None,
        version: int | None = None,
        uuid: str | None = None,
    ):
        info = self._refactor_component_info(info)

        if uuid is None:
            with self.session_context() as session:
                stmt = (
                    self.component_table.update()
                    .where(
                        self.component_table.c.type_id == type_id,
                        self.component_table.c.identifier == identifier,
                        self.component_table.c.version == version,
                    )
                    .values(**info)
                )
                session.execute(stmt)
                self._cache.replace_metadata(
                    type_id=type_id,
                    identifier=identifier,
                    version=version,
                    metadata=info,
                )
        else:
            with self.session_context() as session:
                stmt = (
                    self.component_table.update()
                    .where(self.component_table.c.uuid == uuid)
                    .values(**info)
                )
                session.execute(stmt)
                self._cache.replace_metadata(uuid=uuid, metadata=info)

    def show_cdc_tables(self):
        """Show tables to be consumed with cdc."""
        with self.session_context() as session:
            stmt = select(self.component_table)
            res = self.query_results(self.component_table, stmt, session)
        res = [r['identifier'] for r in res]
        return res

    def set_component_status(self, uuid, status: Status):
        """Set status of component with `status`."""
        with self.session_context() as session:
            value = status if isinstance(status, str) else status.value
            stmt = (
                self.component_table.update()
                .where(
                    self.component_table.c.uuid == uuid,
                )
                .values({'status': value})
            )
            session.execute(stmt)

    def _get_component_status(self, uuid):
        """Get status of component."""
        with self.session_context() as session:
            stmt = (
                select(self.component_table)
                .where(self.component_table.c.uuid == uuid)
                .limit(1)
            )
            res = self.query_results(self.component_table, stmt, session)
            if not res:
                return None
            return res[0]['status']

    def _show_cdcs(self, table):
        """Show all triggers in the database.

        :param type_id: the type of the component
        """
        with self.session_context() as session:
            stmt = select(self.component_table)
            stmt = stmt.where(self.component_table.c.cdc_table == table)
            res = self.query_results(self.component_table, stmt, session)
        return res

    def _show_components(self, type_id: t.Optional[str] = None):
        """Show all components in the database.

        :param type_id: the type of the component
        """
        # TODO: cache it.
        with self.session_context() as session:
            stmt = select(self.component_table)
            if type_id is not None:
                stmt = stmt.where(self.component_table.c.type_id == type_id)
            res = self.query_results(self.component_table, stmt, session)
        return res

    def show_component_versions(self, type_id: str, identifier: str):
        """Show all versions of a component in the database.

        :param type_id: the type of the component
        :param identifier: the identifier of the component
        """
        with self.session_context() as session:
            stmt = select(self.component_table).where(
                self.component_table.c.type_id == type_id,
                self.component_table.c.identifier == identifier,
            )
            res = self.query_results(self.component_table, stmt, session)
            versions = [data['version'] for data in res]
            versions = sorted(set(versions), key=lambda x: versions.index(x))
            return versions

    def _update_object(
        self,
        identifier: str,
        type_id: str,
        key: str,
        value: t.Any,
        version: int,
    ):
        with self.session_context() as session:
            stmt = (
                self.component_table.update()
                .where(
                    self.component_table.c.type_id == type_id,
                    self.component_table.c.identifier == identifier,
                    self.component_table.c.version == version,
                )
                .values({key: value})
            )
            session.execute(stmt)

    # --------------- JOBS -----------------

    def create_job(self, info: t.Union[t.Dict, t.List[t.Dict]]):
        """Create a job with the given info.

        :param info: The information used to create the job
        """
        if 'dependencies' in info:
            info['dependencies'] = json.dumps(info['dependencies'])

        with self.session_context(commit=not self.batched) as session:
            if not self.batched:
                stmt = insert(self.job_table).values(**info)
                session.execute(stmt)
            else:
                self._insert_flush['job'].append(info)

    def get_job(self, job_id: str):
        """Get the job with the given job_id.

        :param job_id: The identifier of the job
        """
        with self.session_context() as session:
            stmt = (
                select(self.job_table).where(self.job_table.c.job_id == job_id).limit(1)
            )
            res = self.query_results(self.job_table, stmt, session)
            return res[0] if res else None

    def show_job_ids(self, uuids: str | None = None, status: str = 'running'):
        """Show all job ids in the database."""
        with self.session_context() as session:
            # Start building the select statement
            stmt = select(self.job_table)

            # If a component_identifier is provided, add a where clause to filter by it
            if uuids is not None:
                stmt = stmt.where(self.job_table.c.uuid.in_(uuids))

            # Execute the query and collect results
            res = self.query_results(self.job_table, stmt, session)
        return [r['job_id'] for r in res]

    def show_jobs(
        self,
        identifier: t.Optional[str] = None,
        type_id: t.Optional[str] = None,
    ):
        """Show all jobs in the database.

        :param component_identifier: the identifier of the component
        :param type_id: the type of the component
        """
        with self.session_context() as session:
            # Start building the select statement
            stmt = select(self.job_table)

            # If a component_identifier is provided, add a where clause to filter by it
            if identifier is not None:
                stmt = stmt.where(
                    and_(
                        self.job_table.c.identifier == identifier,
                        self.job_table.c.type_id == type_id,
                    )
                )

            # Execute the query and collect results
            res = self.query_results(self.job_table, stmt, session)

            return res

    def update_job(self, job_id: str, key: str, value: t.Any):
        """Update the job with the given key and value.

        :param job_id: The identifier of the job
        :param key: The key to update
        :param value: The value to update
        """
        with self.session_context() as session:
            stmt = (
                self.job_table.update()
                .where(self.job_table.c.job_id == job_id)
                .values({key: value})
            )
            session.execute(stmt)

    # --------------- Query ID -----------------

    def disconnect(self):
        """Disconnect the client."""

        # TODO: implement me

    def query_results(self, table, statment, session):
        """Query the database and return the results as a list of row datas.

        :param table: The table object to query, used to derive column names.
        :param statment: The SQL statement to execute.
        :param session: The database session within which the query is executed.
        """
        # Some databases don't support defining statment outside of session
        try:
            result = session.execute(statment)
            columns = [col.name for col in table.columns]
            results = []
            for row in result:
                if len(row) != len(columns):
                    raise ValueError(
                        f'Number of columns in result ({row}) does not match '
                        f'number of columns in table ({columns})'
                    )
                results.append(dict(zip(columns, row)))
        except ProgrammingError:
            # Known ProgrammingErrors:
            # - EmptyResults: Duckdb don't support return empty results
            # - NotExist: SnowFlake returns error if a component does not exist
            return []

        return results
