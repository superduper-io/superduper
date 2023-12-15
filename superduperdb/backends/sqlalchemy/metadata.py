import threading
import typing as t
from contextlib import contextmanager

import click
from sqlalchemy import (
    Column,
    MetaData,
    Table,
    delete,
    insert,
    select,
)
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import sessionmaker

from superduperdb import logging
from superduperdb.backends.base.metadata import MetaDataStore, NonExistentMetadataError
from superduperdb.backends.sqlalchemy.db_helper import get_db_config
from superduperdb.base.serializable import Serializable
from superduperdb.components.component import Component as _Component
from superduperdb.misc.colors import Colors

if t.TYPE_CHECKING:
    from superduperdb.backends.base.query import Select


class SQLAlchemyMetadata(MetaDataStore):
    """
    Abstraction for storing meta-data separately from primary data.

    :param conn: connection to the meta-data store
    :param name: Name to identify DB using the connection
    """

    def __init__(
        self,
        conn: t.Any,
        name: t.Optional[str] = None,
    ):
        self.name = name
        self.conn = conn
        self.dialect = conn.dialect.name
        self._init_tables()

        self._lock = threading.Lock()

    def _init_tables(self):
        # Get the DB config for the given dialect
        DBConfig = get_db_config(self.dialect)

        type_string = DBConfig.type_string
        type_json_as_string = DBConfig.type_json_as_string
        type_json_as_text = DBConfig.type_json_as_text
        type_integer = DBConfig.type_integer
        type_datetime = DBConfig.type_datetime
        type_boolean = DBConfig.type_boolean

        query_id_table_args = DBConfig.query_id_table_args
        job_table_args = DBConfig.job_table_args
        parent_child_association_table_args = (
            DBConfig.parent_child_association_table_args
        )
        component_table_args = DBConfig.component_table_args
        meta_table_args = DBConfig.meta_table_args

        metadata = MetaData()
        self.query_id_table = Table(
            'query_id_table',
            metadata,
            Column('query_id', type_string, primary_key=True),
            Column('query', type_json_as_text),
            Column('model', type_string),
            *query_id_table_args,
        )

        self.job_table = Table(
            'job',
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
            Column('cls', type_string),
            *job_table_args,
        )

        self.parent_child_association_table = Table(
            'parent_child_association',
            metadata,
            Column('parent_id', type_string, primary_key=True),
            Column('child_id', type_string, primary_key=True),
            *parent_child_association_table_args,
        )

        self.component_table = Table(
            'component',
            metadata,
            Column('id', type_string, primary_key=True),
            Column('identifier', type_string),
            Column('version', type_integer),
            Column('hidden', type_boolean),
            Column('type_id', type_string),
            Column('cls', type_string),
            Column('module', type_string),
            Column('dict', type_json_as_text),
            *component_table_args,
        )

        self.meta_table = Table(
            'meta',
            metadata,
            Column('key', type_string, primary_key=True),
            Column('value', type_string),
            *meta_table_args,
        )
        metadata.create_all(self.conn)

    def url(self):
        return self.conn.url + self.name

    def drop(self, force: bool = False):
        """
        Drop the metadata store.
        """
        if not force:
            if not click.confirm(
                f'{Colors.RED}[!!!WARNING USE WITH CAUTION AS YOU '
                f'WILL LOSE ALL DATA!!!]{Colors.RESET} '
                'Are you sure you want to drop all meta-data? ',
                default=False,
            ):
                logging.warn('Aborting...')
        self.query_id_table.drop(self.conn)
        self.job_table.drop(self.conn)
        self.parent_child_association_table.drop(self.conn)
        self.component_table.drop(self.conn)
        self.meta_table.drop(self.conn)

    @contextmanager
    def session_context(self):
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

    def component_version_has_parents(
        self, type_id: str, identifier: str, version: int
    ):
        unique_id = _Component.make_unique_id(type_id, identifier, version)
        with self.session_context() as session:
            stmt = (
                select(self.parent_child_association_table)
                .where(
                    self.parent_child_association_table.c.child_id == unique_id,
                )
                .limit(1)
            )
            res = self.query_results(self.parent_child_association_table, stmt, session)
            return len(res) > 0

    def create_component(self, info: t.Dict):
        if 'hidden' not in info:
            info['hidden'] = False
        info['id'] = f'{info["type_id"]}/{info["identifier"]}/{info["version"]}'
        with self.session_context() as session:
            stmt = insert(self.component_table).values(**info)
            session.execute(stmt)

    def create_parent_child(self, parent_id: str, child_id: str):
        with self.session_context() as session:
            stmt = insert(self.parent_child_association_table).values(
                parent_id=parent_id, child_id=child_id
            )
            session.execute(stmt)

    def delete_component_version(self, type_id: str, identifier: str, version: int):
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

    def _get_component(
        self,
        type_id: str,
        identifier: str,
        version: int,
        allow_hidden: bool = False,
    ):
        with self.session_context() as session:
            stmt = select(self.component_table).where(
                self.component_table.c.type_id == type_id,
                self.component_table.c.identifier == identifier,
                self.component_table.c.version == version,
            )
            if not allow_hidden:
                stmt = stmt.where(self.component_table.c.hidden == allow_hidden)

            res = self.query_results(self.component_table, stmt, session)
            return res[0] if res else None

    def get_component_version_parents(self, unique_id: str):
        with self.session_context() as session:
            stmt = select(self.parent_child_association_table).where(
                self.parent_child_association_table.c.child_id == unique_id,
            )
            res = self.query_results(self.parent_child_association_table, stmt, session)
            parents = [r['parent_id'] for r in res]
            return parents

    def get_latest_version(
        self, type_id: str, identifier: str, allow_hidden: bool = False
    ):
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

    def replace_component(
        self,
        info: t.Dict[str, t.Any],
        identifier: str,
        type_id: str,
        version: t.Optional[int] = None,
    ) -> None:
        if version is not None:
            version = self.get_latest_version(type_id, identifier)
        return self._replace_object(
            info=info,
            identifier=identifier,
            type_id=type_id,
            version=version,
        )

    def show_components(self, type_id: t.Optional[str] = None, **kwargs):
        with self.session_context() as session:
            stmt = select(self.component_table)
            if type_id is not None:
                stmt = stmt.where(self.component_table.c.type_id == type_id)
            res = self.query_results(self.component_table, stmt, session)
        identifiers = [data['identifier'] for data in res]
        identifiers = sorted(set(identifiers), key=lambda x: identifiers.index(x))
        return identifiers

    def show_component_versions(self, type_id: str, identifier: str):
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
        with self.session_context() as session:
            stmt = insert(self.job_table).values(**info)
            session.execute(stmt)

    def get_job(self, job_id: str):
        with self.session_context() as session:
            stmt = (
                select(self.job_table)
                .where(self.job_table.c.identifier == job_id)
                .limit(1)
            )
            res = self.query_results(self.job_table, stmt, session)
            return res[0] if res else None

    def listen_job(self, identifier: str):
        # Not supported currently
        raise NotImplementedError

    def show_jobs(self):
        with self.session_context() as session:
            stmt = select(self.job_table)
            res = self.query_results(self.job_table, stmt, session)
            return [r['identifier'] for r in res]

    def update_job(self, job_id: str, key: str, value: t.Any):
        with self.session_context() as session:
            stmt = (
                self.job_table.update()
                .where(self.job_table.c.identifier == job_id)
                .values({key: value})
            )
            session.execute(stmt)

    def write_output_to_job(self, identifier, msg, stream):
        # Not supported currently
        raise NotImplementedError

    # --------------- METADATA -----------------

    def create_metadata(self, key, value):
        with self.session_context() as session:
            stmt = insert(self.meta_table).values(key=key, value=value)
            session.execute(stmt)

    def get_metadata(self, key):
        with self.session_context() as session:
            stmt = select(self.meta_table).where(self.meta_table.c.key == key).limit(1)
            res = self.query_results(self.meta_table, stmt, session)
            value = res[0]['value'] if res else None
            return value

    def update_metadata(self, key, value):
        with self.session_context() as session:
            stmt = (
                self.meta_table.update()
                .where(self.meta_table.c.key == key)
                .values({key: value})
            )
            session.execute(stmt)

    # --------------- Query ID -----------------
    def add_query(self, query: 'Select', model: str):
        query_hash = str(hash(query))

        with self.session_context() as session:
            with self._lock:
                row = {
                    'query': query.serialize(),
                    'model': model,
                    'query_id': query_hash,
                }

            stmt = insert(self.query_id_table).values(**row)
            session.execute(stmt)

    def get_query(self, query_hash: str):
        """
        Get the query from the query table corresponding to the query hash
        """
        try:
            with self.session_context() as session:
                stmt = (
                    select(self.query_id_table)
                    .where(self.query_id_table.c.query_id == str(query_hash))
                    .limit(1)
                )
                res = self.query_results(self.query_id_table, stmt, session)
                out = res[0] if res else None
        except AttributeError as e:
            if 'NoneType' in str(e):
                raise NonExistentMetadataError(
                    f'Query hash {query_hash} does not exist'
                )
            raise e

        if out is None:
            raise NonExistentMetadataError(f'Query hash {query_hash} does not exist')

    def get_model_queries(self, model: str):
        """
        Get queries related to the given model.
        """
        with self.session_context() as session:
            stmt = select(self.query_id_table).where(
                self.query_id_table.c.model == model
            )
            queries = self.query_results(self.query_id_table, stmt, session)

            unpacked_queries = []
            for row in queries:
                id = row['query_id']
                serialized = row['query']
                query = Serializable.deserialize(serialized)
                unpacked_queries.append(
                    {'query_id': id, 'query': query, 'sql': query.repr_()}
                )
            return unpacked_queries

    def disconnect(self):
        """
        Disconnect the client
        """

        # TODO: implement me

    def query_results(self, table, statment, session):
        # Some databases don't support defining statment outside of session
        result = session.execute(statment)
        columns = [col.name for col in table.columns]
        results = []
        try:
            for row in result:
                if len(row) != len(columns):
                    raise ValueError(
                        f'Number of columns in result ({row}) does not match '
                        f'number of columns in table ({columns})'
                    )
                results.append(dict(zip(columns, row)))
        except ProgrammingError:
            # Some databases don't support return empty results, such as duckdb
            return []

        return results
