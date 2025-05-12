import threading
import time
import typing as t
import uuid
from contextlib import contextmanager

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


class ThreadLocalConnectionManager:
    """A thread-local connection manager for ibis."""

    def __init__(self, uri: str, flavour: t.Optional[str] = None):
        """Initialize the thread-local connection manager.

        :param uri: URI to the database.
        :param flavour: Flavour of the database.
        """
        self.uri = uri
        self.flavour = flavour
        self.local = threading.local()
        self.lock = threading.RLock()

    def _create_connection(self):
        """Create a new connection specifically for this thread."""
        name = self.uri.split("//")[0]
        in_memory = False
        ibis_conn = ibis.connect(self.uri)
        return ibis_conn, name, in_memory

    @contextmanager
    def get_connection(self):
        """Get a connection for the current thread, creating it if it doesn't exist."""
        if not hasattr(self.local, "connection"):
            with self.lock:  # Lock only during connection creation
                self.local.connection, self.local.name, self.local.in_memory = (
                    self._create_connection()
                )
                logging.info(
                    f"Created new connection for thread"
                    f"'{threading.current_thread().name}'"
                )

        try:
            logging.debug(
                f"Reusing connection for thread '{threading.current_thread().name}'"
            )
            yield self.local.connection
        except Exception as e:
            # If there's a connection error,
            # clear the thread's connection so a new one will be created next time
            if hasattr(self.local, "connection"):
                delattr(self.local, "connection")
            logging.error(
                f"Connection error in thread '{threading.current_thread().name}: {e}'"
            )
            raise e


class IbisDataBackend(BaseDataBackend):
    """Ibis data backend for the database.

    :param uri: URI to the databackend database.
    :param flavour: Flavour of the databackend.
    """

    def __init__(self, uri: str, plugin: t.Any, flavour: t.Optional[str] = None):
        super().__init__(uri=uri, flavour=flavour, plugin=plugin)
        self.uri = uri
        self.connection_manager = ThreadLocalConnectionManager(uri, flavour)

        # Get a connection to initialize
        self.reconnect()

    def reconnect(self):
        """Reconnect to the database client."""
        with self.connection_manager.get_connection() as conn:
            self.dialect = getattr(conn, "name", "base")
            self.db_helper = get_db_helper(self.dialect)

    def random_id(self):
        """Generate a random ID."""
        return str(uuid.uuid4())

    def to_id(self, id):
        """Convert the ID to a string."""
        return str(id)

    def url(self):
        """Get the URL of the database."""
        with self.connection_manager.get_connection() as conn:
            return conn.con.url + self.name

    def build_artifact_store(self):
        """Build artifact store for the database."""
        return FileSystemArtifactStore(conn=".superduper/artifacts/", name="ibis")

    def drop_table(self, table):
        """Drop the outputs."""
        with self.connection_manager.get_connection() as conn:
            conn.drop_table(table)

    def check_output_dest(self, predict_id) -> bool:
        """Check if the output destination exists.

        :param predict_id: The identifier of the prediction.
        """
        with self.connection_manager.get_connection() as conn:
            try:
                conn.table(f"{CFG.output_prefix}{predict_id}")
                return True
            except (NoSuchTableError, ibis.IbisError, TableNotFound):
                return False

    def create_table_and_schema(self, identifier: str, schema: Schema, primary_id: str):
        """Create a schema in the data-backend.

        :param identifier: The identifier of the table.
        :param mapping: The mapping of the schema.
        """
        with self.connection_manager.get_connection() as conn:
            mapping = convert_schema_to_fields(schema)
            if primary_id not in mapping:
                mapping[primary_id] = "string"
            try:
                mapping = self.db_helper.process_schema_types(mapping)
                t = self._create_table_with_retry(
                    conn, identifier, schema=ibis.schema(mapping)
                )
            except Exception as e:
                if "exists" in str(e) or "override" in str(e):
                    logging.warn("Table already exists, skipping...")
                    t = conn.table(identifier)
                else:
                    raise e

            return t

    def _create_table_with_retry(self, conn, table_name, schema, retry=3):
        for attempt in range(retry):
            t = conn.create_table(table_name, schema=schema)

            all_tables = conn.list_tables()
            if table_name in all_tables:
                return t
            else:
                logging.warn(
                    f"Failed to create table {table_name}. "
                    f"Attempt {attempt + 1}/{retry}"
                )
                time.sleep(1)

            raise exceptions.NotFound("table", table_name)

    def drop(self, force: bool = False):
        """Drop tables or collections in the database.

        :param force: Whether to force the drop.
        """
        if not force and not click.confirm("Are you sure you want to drop all tables?"):
            logging.info("Aborting drop tables")
            return

        with self.connection_manager.get_connection() as conn:
            for table in conn.list_tables():
                logging.info(f"Dropping table: {table}")
                self.drop_table(table)

    def get_table(self, identifier):
        """Get a table or collection from the database.

        :param identifier: The identifier of the table or collection.
        """
        with self.connection_manager.get_connection() as conn:
            try:
                return conn.table(identifier)
            except ibis.common.exceptions.IbisError as e:
                raise exceptions.NotFound("table", identifier) from e

    def disconnect(self):
        """Disconnect the client."""
        # TODO: implement me

    def list_tables(self):
        """List all tables or collections in the database."""
        with self.connection_manager.get_connection() as conn:
            return conn.list_tables()

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
        primary_id = self.primary_id(table)
        for r in documents:
            if primary_id not in r:
                r[primary_id] = str(uuid.uuid4())

        documents = pandas.DataFrame(documents)
        documents = documents.dropna(axis=1, how="all")
        documents = documents.to_dict(orient="records")
        ids = [r[primary_id] for r in documents]

        # Convert the documents to a memtable with the correct schema
        with self.connection_manager.get_connection() as conn:
            schema = conn.table(table).schema()
            memtable = ibis.memtable(documents, schema=schema)
            conn.insert(table, memtable)
            return ids

    def missing_outputs(self, query, predict_id: str) -> t.List[str]:
        """Get missing outputs from the database."""
        with self.connection_manager.get_connection() as conn:
            pid = self.primary_id(query.table)
            query = self._build_native_query(conn, query)
            output_table = conn.table(f"{CFG.output_prefix}{predict_id}")
            q = query.anti_join(output_table, output_table["_source"] == query[pid])
            rows = q.execute().to_dict(orient="records")
            return [r[pid] for r in rows]

    def select(self, query):
        """Select data from the database."""
        with self.connection_manager.get_connection() as conn:
            native_query = self._build_native_query(conn, query)
            return native_query.execute().to_dict(orient="records")

    def _build_native_query(self, conn, query):
        try:
            q = conn.table(query.table)
        except TableNotFound as e:
            raise exceptions.NotFound("table", query.table) from e

        pid = None
        predict_ids = (
            query.decomposition.outputs.args if query.decomposition.outputs else []
        )

        for part in query.parts:
            if isinstance(part, QueryPart) and part.name != "outputs":
                args = []
                for a in part.args:
                    if isinstance(a, Query) and str(a).endswith(".primary_id"):
                        args.append(self.primary_id(query.table))
                    elif isinstance(a, Query):
                        args.append(self._build_native_query(conn, a))
                    else:
                        args.append(a)

                kwargs = {}
                for k, v in part.kwargs.items():
                    if isinstance(a, Query) and str(a).endswith(".primary_id"):
                        args.append(self.primary_id(query.table))
                    elif isinstance(v, Query):
                        kwargs[k] = self._build_native_query(conn, v)
                    else:
                        kwargs[k] = v

                if part.name == "select" and len(args) == 0:
                    pass

                else:
                    if part.name == "select" and predict_ids and args:
                        args.extend(
                            [
                                f"{CFG.output_prefix}{pid}"
                                for pid in predict_ids
                                if f"{CFG.output_prefix}{pid}" not in args
                            ]
                        )
                        args = list(set(args))
                    q = getattr(q, part.name)(*args, **kwargs)

            elif isinstance(part, QueryPart) and part.name == "outputs":
                if pid is None:
                    pid = self.primary_id(query.table)

                original_q = q
                for predict_id in part.args:
                    output_t = conn.table(f"{CFG.output_prefix}{predict_id}").select(
                        f"{CFG.output_prefix}{predict_id}", "_source"
                    )
                    q = q.join(output_t, output_t["_source"] == original_q[pid])

            elif isinstance(part, str):
                if part == "primary_id":
                    if pid is None:
                        pid = self.primary_id(query.table)
                    part = pid
                q = q[part]
            else:
                raise ValueError(f"Unknown query part: {part}")
        return q

    def execute_native(self, query: str):
        """Execute a native SQL query.

        :param query: The query to execute.
        """
        with self.connection_manager.get_connection() as conn:
            return pandas.DataFrame(conn.raw_sql(query)).to_dict(orient="records")


class SQLDatabackend(IbisDataBackend):
    """SQL data backend for the database."""

    def __init__(self, uri, plugin, flavour=None):
        super().__init__(uri, plugin, flavour)
        self._create_sqlalchemy_engine()
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

    def _create_sqlalchemy_engine(self):
        with self.connection_manager.get_connection() as conn:
            self.alchemy_engine = create_engine(self.uri, creator=lambda: conn.con)
            if not self._test_engine():
                logging.warn(
                    "Unable to reuse the ibis connection "
                    "to create the SQLAlchemy engine. "
                    "Creating a new connection with the URI."
                )
                self.alchemy_engine = create_engine(self.uri)

    def _test_engine(self):
        """Test the engine."""
        try:
            with self.alchemy_engine.connect() as conn:
                if not conn.closed:
                    return True

            return False

        except Exception as e:
            logging.debug(f"Error testing engine: {e}")
            return False
