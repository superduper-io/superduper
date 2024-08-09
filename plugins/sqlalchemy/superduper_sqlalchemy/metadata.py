import threading
import typing as t
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
from superduper.backends.sqlalchemy.db_helper import get_db_config
from superduper.misc.colors import Colors


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

    def reconnect(self):
        """Reconnect to sqlalchmey metadatastore."""
        sql_conn = create_engine(self.uri)
        self.conn = sql_conn

        # TODO: is it required to init after
        # a reconnect.
        self._init_tables()

    def _init_tables(self):
        # Get the DB config for the given dialect
        DBConfig = get_db_config(self.dialect)

        type_string = DBConfig.type_string
        type_json_as_string = DBConfig.type_json_as_string
        type_json_as_text = DBConfig.type_json_as_text
        type_integer = DBConfig.type_integer
        type_datetime = DBConfig.type_datetime
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
            Column('identifier', type_string, primary_key=True),
            Column('component_identifier', type_string),
            Column('type_id', type_string),
            Column('info', type_json_as_string),
            Column('time', type_datetime),
            Column('status', type_string),
            Column('args', type_json_as_string),
            Column('kwargs', type_json_as_text),
            Column('method_name', type_string),
            Column('stdout', type_json_as_string),
            Column('stderr', type_json_as_string),
            Column('_path', type_string),
            Column('job_id', type_string),
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
            'COMPONENT',
            metadata,
            Column('id', type_string, primary_key=True),
            Column('identifier', type_string),
            Column('version', type_integer),
            Column('hidden', type_boolean),
            Column('type_id', type_string),
            Column('_path', type_string),
            Column('dict', type_json_as_text),
            *component_table_args,
        )

        metadata.create_all(self.conn)

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

    @contextmanager
    def session_context(self):
        """Provide a transactional scope around a series of operations."""
        sm = sessionmaker(bind=self.conn)
        session = sm()
        try:
            yield session
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

    def create_component(self, info: t.Dict):
        """Create a component in the metadata store.

        :param info: the information to create the component
        """
        new_info = self._refactor_component_info(info)
        with self.session_context() as session:
            stmt = insert(self.component_table).values(**new_info)
            session.execute(stmt)

    def delete_parent_child(self, parent_id: str, child_id: str):
        """
        Delete parent-child relationships between two components.

        :param parent: parent component uuid
        :param child: child component uuid
        """
        with self.session_context() as session:
            stmt = delete(self.parent_child_association_table).where(
                self.parent_child_association_table.c.parent_id == parent_id,
                self.parent_child_association_table.c.child_id == child_id,
            )
            session.execute(stmt)

    def create_parent_child(self, parent_id: str, child_id: str):
        """Create a parent-child relationship between two components.

        :param parent_id: the parent component
        :param child_id: the child component
        """
        with self.session_context() as session:
            stmt = insert(self.parent_child_association_table).values(
                parent_id=parent_id, child_id=child_id
            )
            session.execute(stmt)

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

    def get_component_by_uuid(self, uuid: str, allow_hidden: bool = False):
        """Get a component by UUID.

        :param uuid: UUID of component
        :param allow_hidden: whether to load hidden components
        """
        with self.session_context() as session:
            stmt = (
                select(self.component_table)
                .where(
                    self.component_table.c.id == uuid,
                )
                .limit(1)
            )
            res = self.query_results(self.component_table, stmt, session)
            try:
                r = res[0]
            except IndexError:
                raise NonExistentMetadataError(
                    f'Table with uuid: {uuid} does not exist'
                )

        return self._get_component(
            type_id=r['type_id'],
            identifier=r['identifier'],
            version=r['version'],
            allow_hidden=allow_hidden,
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
                dict_ = res['dict']
                del res['dict']
                res = {**res, **dict_}
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
        component_fields = ['identifier', 'version', 'hidden', 'type_id', '_path']
        new_info = {k: info[k] for k in component_fields}
        new_info['dict'] = {k: info[k] for k in info if k not in component_fields}
        new_info['id'] = new_info['dict']['uuid']
        return new_info

    def get_latest_version(
        self, type_id: str, identifier: str, allow_hidden: bool = False
    ):
        """Get the latest version of a component.

        :param type_id: the type of the component
        :param identifier: the identifier of the component
        :param allow_hidden: whether to allow hidden components
        """
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
            res = session.execute(stmt)
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

    def _replace_object(self, info, identifier, type_id, version):
        info = self._refactor_component_info(info)
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

    def _show_components(self, type_id: t.Optional[str] = None):
        """Show all components in the database.

        :param type_id: the type of the component
        """
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

    def create_job(self, info: t.Dict):
        """Create a job with the given info.

        :param info: The information used to create the job
        """
        with self.session_context() as session:
            stmt = insert(self.job_table).values(**info)
            session.execute(stmt)

    def get_job(self, job_id: str):
        """Get the job with the given job_id.

        :param job_id: The identifier of the job
        """
        with self.session_context() as session:
            stmt = (
                select(self.job_table)
                .where(self.job_table.c.identifier == job_id)
                .limit(1)
            )
            res = self.query_results(self.job_table, stmt, session)
            return res[0] if res else None

    def show_jobs(
        self,
        component_identifier: t.Optional[str] = None,
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
            if component_identifier is not None:
                stmt = stmt.where(
                    and_(
                        self.job_table.c.component_identifier == component_identifier,
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
                .where(self.job_table.c.identifier == job_id)
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
