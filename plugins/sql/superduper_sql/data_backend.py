import time
import typing as t
import uuid

import click
import ibis
import pandas
from ibis.common.exceptions import TableNotFound
from sqlalchemy import MetaData, Table, create_engine
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.orm import sessionmaker
from superduper import CFG, logging
from superduper.backends.base.data_backend import BaseDataBackend
from superduper.base import exceptions
from superduper.base.artifacts import FileSystemArtifactStore
from superduper.base.query import Query, QueryPart
from superduper.base.schema import Schema

from superduper_sql.db_helper import get_db_helper
from superduper_sql.utils import convert_schema_to_fields

BASE64_PREFIX = "base64:"
# TODO make this a global variable in main project
INPUT_KEY = "_source"


def _connection_callback(uri, flavour):
    name = uri.split("//")[0]
    in_memory = False
    ibis_conn = ibis.connect(uri)
    return ibis_conn, name, in_memory


class IbisDataBackend(BaseDataBackend):
    """Ibis data backend for the database.

    :param uri: URI to the databackend database.
    :param flavour: Flavour of the databackend.
    """

    def __init__(self, uri: str, plugin: t.Any, flavour: t.Optional[str] = None):
        self.connection_callback = lambda: _connection_callback(uri, flavour)
        conn, name, in_memory = self.connection_callback()
        super().__init__(uri=uri, flavour=flavour, plugin=plugin)
        self.conn = conn
        self.name = name
        self.in_memory = in_memory
        self.overwrite = False
        self._setup(conn)

    def random_id(self):
        """Generate a random ID."""
        return str(uuid.uuid4())

    def to_id(self, id):
        """Convert the ID to a string."""
        return str(id)

    def _setup(self, conn):
        self.dialect = getattr(conn, "name", "base")
        self.db_helper = get_db_helper(self.dialect)

    def reconnect(self):
        """Reconnect to the database client."""
        conn, _, _ = self.connection_callback()
        self.conn = conn
        self._setup(conn)

    def url(self):
        """Get the URL of the database."""
        return self.conn.con.url + self.name

    def build_artifact_store(self):
        """Build artifact store for the database."""
        return FileSystemArtifactStore(conn=".superduper/artifacts/", name="ibis")

    def drop_table(self, table):
        """Drop the outputs."""
        self.conn.drop_table(table)

    def check_output_dest(self, predict_id) -> bool:
        """Check if the output destination exists.

        :param predict_id: The identifier of the prediction.
        """
        try:
            self.conn.table(f"{CFG.output_prefix}{predict_id}")
            return True
        except (NoSuchTableError, ibis.IbisError, TableNotFound):
            return False

    def create_table_and_schema(self, identifier: str, schema: Schema, primary_id: str):
        """Create a schema in the data-backend.

        :param identifier: The identifier of the table.
        :param mapping: The mapping of the schema.
        """
        mapping = convert_schema_to_fields(schema)
        if primary_id not in mapping:
            mapping[primary_id] = "string"
        try:
            mapping = self.db_helper.process_schema_types(mapping)
            t = self._create_table_with_retry(identifier, schema=ibis.schema(mapping))
        except Exception as e:
            if "exists" in str(e) or "override" in str(e):
                logging.warn("Table already exists, skipping...")
                t = self.conn.table(identifier)
            else:
                raise e

        return t

    def _create_table_with_retry(self, table_name, schema, retry=3):
        for attempt in range(retry):
            t = self.conn.create_table(table_name, schema=schema)

            if table_name in self.conn.list_tables():
                return t
            else:
                logging.warn(
                    f"Failed to create table {table_name}. "
                    f"Attempt {attempt + 1}/{retry}"
                )
                time.sleep(1)

        raise exceptions.TableNotFoundError(f"Failed to create table {table_name}")

    def drop(self, force: bool = False):
        """Drop tables or collections in the database.

        :param force: Whether to force the drop.
        """
        if not force and not click.confirm("Are you sure you want to drop all tables?"):
            logging.info("Aborting drop tables")
            return

        for table in self.conn.list_tables():
            logging.info(f"Dropping table: {table}")
            self.drop_table(table)

    def get_table(self, identifier):
        """Get a table or collection from the database.

        :param identifier: The identifier of the table or collection.
        """
        try:
            return self.conn.table(identifier)
        except ibis.common.exceptions.IbisError:
            raise exceptions.TableNotFoundError(
                f'Table {identifier} not found in database'
            )

    def disconnect(self):
        """Disconnect the client."""
        # TODO: implement me

    def list_tables(self):
        """List all tables or collections in the database."""
        return self.conn.list_tables()

    def replace(self, table: str, condition: t.Dict, r: t.Dict) -> t.List[str]:
        """Replace data.

        :param table: The table to insert into.
        :param condition: The condition to replace.
        :param r: The data to replace.
        """
        raise NotImplementedError

    def update(self, table, condition, key, value):
        """Update data in the database."""
        raise NotImplementedError

    def delete(self, table, condition):
        """Delete data from the database."""
        raise NotImplementedError

    def insert(self, table, documents):
        """Insert data into the database."""
        primary_id = self.primary_id(self.db[table])
        for r in documents:
            if primary_id not in r:
                r[primary_id] = str(uuid.uuid4())

        documents = pandas.DataFrame(documents)
        documents = documents.dropna(axis=1, how='all')
        documents = documents.to_dict(orient='records')
        ids = [r[primary_id] for r in documents]
        self.conn.insert(table, documents)
        return ids

    def missing_outputs(self, query, predict_id: str) -> t.List[str]:
        """Get missing outputs from the database."""
        pid = self.primary_id(query)
        query = self._build_native_query(query)
        output_table = self.conn.table(f"{CFG.output_prefix}{predict_id}")
        q = query.anti_join(output_table, output_table['_source'] == query[pid])
        rows = q.execute().to_dict(orient='records')
        return [r[pid] for r in rows]

    def primary_id(self, query):
        """Get the primary ID of the query."""
        return self.db.metadata.get_component(
            component='Table', identifier=query.table
        )['primary_id']

    def select(self, query):
        """Select data from the database."""
        native_query = self._build_native_query(query)
        return native_query.execute().to_dict(orient='records')

    def _build_native_query(self, query):
        try:
            q = self.conn.table(query.table)
        except ibis.IbisError as e:
            if 'not found' in str(e):
                raise exceptions.DatabackendError(f'Table {query.table} not found')
            raise e
        pid = None
        predict_ids = (
            query.decomposition.outputs.args if query.decomposition.outputs else []
        )

        for part in query.parts:
            if isinstance(part, QueryPart) and part.name != 'outputs':
                args = []
                for a in part.args:
                    if isinstance(a, Query) and str(a).endswith('.primary_id'):
                        args.append(self.primary_id(query))
                    elif isinstance(a, Query):
                        args.append(self._build_native_query(a))
                    else:
                        args.append(a)

                kwargs = {}
                for k, v in part.kwargs.items():
                    if isinstance(a, Query) and str(a).endswith('.primary_id'):
                        args.append(self.primary_id(query))
                    elif isinstance(v, Query):
                        kwargs[k] = self._build_native_query(v)
                    else:
                        kwargs[k] = v

                if part.name == 'select' and len(args) == 0:
                    pass

                else:
                    if part.name == 'select' and predict_ids and args:
                        args.extend(
                            [
                                f'{CFG.output_prefix}{pid}'
                                for pid in predict_ids
                                if f'{CFG.output_prefix}{pid}' not in args
                            ]
                        )
                        args = list(set(args))
                    q = getattr(q, part.name)(*args, **kwargs)

            elif isinstance(part, QueryPart) and part.name == 'outputs':
                if pid is None:
                    pid = self.primary_id(query)

                original_q = q
                for predict_id in part.args:
                    output_t = self.conn.table(
                        f"{CFG.output_prefix}{predict_id}"
                    ).select(f"{CFG.output_prefix}{predict_id}", "_source")
                    q = q.join(output_t, output_t['_source'] == original_q[pid])

            elif isinstance(part, str):
                if part == 'primary_id':
                    if pid is None:
                        pid = self.primary_id(query)
                    part = pid
                q = q[part]
            else:
                raise ValueError(f'Unknown query part: {part}')
        return q

    def execute_native(self, query: str):
        """Execute a native SQL query.

        :param query: The query to execute.
        """
        return pandas.DataFrame(self.conn.raw_sql(query)).to_dict(orient='records')


class SQLDatabackend(IbisDataBackend):
    """SQL data backend for the database."""

    def __init__(self, uri, plugin, flavour=None):
        super().__init__(uri, plugin, flavour)
        self.alchemy_engine = create_engine(uri, creator=lambda: self.conn.con)
        self.sm = sessionmaker(bind=self.alchemy_engine)

    def update(self, table, condition, key, value):
        """Update data in the database."""
        with self.sm() as session:
            metadata = MetaData()

            metadata.reflect(bind=session.bind)
            table = Table(table, metadata, autoload_with=session.bind)

            stmt = table.update()
            for c, v in condition.items():
                stmt = stmt.where(table.c[c] == v)

            stmt = stmt.values(**{key: value})

            session.execute(stmt)
            session.commit()
            session.commit()

    def replace(self, table, condition, r):
        """Replace data."""
        with self.sm() as session:
            metadata = MetaData()

            metadata.reflect(bind=session.bind)
            table = Table(table, metadata, autoload_with=session.bind)

            stmt = table.update()
            for c, v in condition.items():
                stmt = stmt.where(table.c[c] == v)

            stmt = stmt.values(**r)

            session.execute(stmt)
            session.commit()
            session.commit()

    def delete(self, table, condition):
        """Delete data from the database."""
        with self.sm() as session:
            metadata = MetaData()

            metadata.reflect(bind=session.bind)
            table = Table(table, metadata, autoload_with=session.bind)

            stmt = table.delete()

            for col_name, value in condition.items():
                stmt = stmt.where(table.c[col_name] == value)

            session.execute(stmt)
            session.commit()
            session.commit()
