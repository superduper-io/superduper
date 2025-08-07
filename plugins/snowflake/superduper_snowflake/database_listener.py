import os
import re
import threading
import time
import typing as t
import uuid

from snowflake.core import Root
from snowflake.core.stream import Stream, StreamSourceTable
from snowflake.snowpark import Session
from superduper import CFG, logging
from superduper.backends.base.cdc import BaseDatabaseListener, DBEvent
from superduper.misc.threading import Event

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


class _SnowflakeCDCChangeStream:
    def __init__(self, session, name, table):
        self.session = session
        self.name = name
        self.table = table

    def reset_stream(self):
        self.session.sql(
            f'CREATE OR REPLACE STREAM {self.name} ON TABLE "{self.table}";'
        ).collect()

    def read_stream(self):
        temp_table = f"TEMP_{self.name}_{uuid.uuid4().hex[:8]}"

        try:
            self.session.sql(
                f"""
                CREATE TEMPORARY TABLE {temp_table} AS
                SELECT * FROM {self.name}
                WHERE METADATA$ACTION = 'INSERT'
            """
            ).collect()

            stream_data = self.session.sql(f"SELECT * FROM {temp_table}").collect()
            return stream_data
        finally:
            self.session.sql(f"DROP TABLE IF EXISTS {temp_table}").collect()

    def next(self):
        stream_data = []
        try:
            stream_data = self.read_stream()
            if stream_data:
                logging.info(f"Stream data: {stream_data}")

        except Exception as e:
            import traceback

            logging.error(traceback.format_exc())
            # Recreate table
            if "Base table" in str(e) and "dropped" in str(e):
                logging.warn(f"Stream {self.name} was dropped, recreating...")
                stream_data = self.reset_stream()
            else:
                raise

        return stream_data or []


class SnowflakeDatabaseListener(BaseDatabaseListener):
    """Snowflake specific database listener implementation.

    :param db: A superduper instance
    :param table: A table or collection on which the listener is invoked
    :param stop_event: A threading event flag to notify for stoppage
    :param timeout: A timeout for the listener
    :param error_handler: A callable to handle errors during listening
    """

    _scheduler: t.Optional[threading.Thread]

    def __init__(
        self,
        db: "Datalayer",
        table: str,
        stop_event: Event,
        timeout: t.Optional[float] = None,
        error_handler: t.Optional[t.Callable] = None,
    ):
        self.primary_id = db[table].primary_id.execute()

        self.session = None
        self.table = table.replace("-", "_")
        self.stream_name = f'STREAM_{table}'.upper()

        super().__init__(
            db=db,
            table=table,
            stop_event=stop_event,
            timeout=timeout,
            error_handler=error_handler,
        )

    def on_create(
        self,
        ids: t.Sequence,
        db: "Datalayer",
        collection: str,
    ) -> None:
        """Handle creation of new records in the database.

        :param ids: Document ids
        :param db: A superduper instance
        :param collection: The collection on which change was observed
        """
        logging.info(f"Triggered `on_create` handler with ids {ids}.")
        self.create_event(ids=ids, db=db, table=collection, event=DBEvent.insert)

    def on_update(
        self,
        ids: t.Sequence,
        db: "Datalayer",
        collection: str,
    ) -> None:
        """
        Handle updates to existing records in the database.

        :param ids: Document ids
        :param db: A superduper instance
        :param collection: The collection on which change was observed
        """
        logging.info(f"Triggered `on_update` handler with ids {ids}.")

        self.create_event(ids=ids, db=db, table=collection, event=DBEvent.update)

    def on_delete(
        self,
        ids: t.Sequence,
        db: "Datalayer",
        collection: str,
    ) -> None:
        """Handle deletion of records in the database.

        :param ids: Document ids
        :param db: A superduper instance
        :param collection: The collection on which change was observed
        """
        logging.debug("Triggered `on_delete` handler.")
        self.create_event(ids=ids, db=db, table=collection, event=DBEvent.delete)

    def setup_cdc(self) -> _SnowflakeCDCChangeStream:
        """Set up the CDC listener for Snowflake."""
        uri = CFG.data_backend

        # TODO use helper function from ibis

        schema = None
        database = None
        if uri == "snowflake://":
            host = os.environ["SNOWFLAKE_HOST"]
            port = int(os.environ["SNOWFLAKE_PORT"])
            account = os.environ["SNOWFLAKE_ACCOUNT"]
            token = open("/snowflake/session/token").read()
            warehouse = os.environ["SNOWFLAKE_WAREHOUSE"]
            database = os.environ["SNOWFLAKE_DATABASE"]
            schema = os.environ["SUPERDUPER_DATA_SCHEMA"]

            connection_parameters = {
                "token": token,
                "account": account,
                "database": database,
                "schema": schema,
                "warehouse": warehouse,
                "authenticator": "oauth",
                "port": port,
                "host": host,
            }
        else:
            pattern = r"snowflake://(?P<user>[^:]+):(?P<password>[^@]+)@(?P<account>[^/]+)/(?P<database>[^/]+)/(?P<schema>[^/]+)"
            match = re.match(pattern, uri)
            schema = match.group("schema")
            database = match.group("database")
            if match:
                connection_parameters = {
                    "user": match.group("user"),
                    "password": match.group("password"),
                    "account": match.group("account"),
                    "database": match.group("database"),
                    "schema": match.group("schema"),
                    # TODO: check warehouse
                    "warehouse": "base",
                }

            else:
                raise ValueError(f"URI `{uri}` is invalid!")

        logging.info("Creating Snowpark session")
        self.session = Session.builder.configs(connection_parameters).create()
        logging.info("Creating Snowpark session... DONE")

        # check if exists
        assert schema
        assert database

        root = Root(self.session)

        streams = root.databases[database].schemas[schema].streams.iter()
        exists = False

        for stream in streams:
            if stream.name == self.stream_name:
                logging.warn(f"CDC stream already exists for name {self.stream_name}")
                exists = True
                break

        if not exists:
            stream_on_table = Stream(
                name=self.stream_name,
                stream_source=StreamSourceTable(
                    name=f'"{self.table}"',
                    append_only=True,
                    show_initial_rows=False,
                ),
                comment="create stream on table",
            )
            streams = root.databases[database].schemas[schema].streams

            logging.info(f"Creating stream with name {self.stream_name}...")
            streams.create(stream_on_table)
            logging.info(f"Creating stream with name {self.stream_name} DONE")

        logging.info("Start to listen to database with identity ", self.identifier)
        stream_iterator = _SnowflakeCDCChangeStream(
            session=self.session,
            name=self.stream_name,
            table=self.table,
        )
        logging.info("Start to listen database with identity... DONE", self.identifier)

        return stream_iterator

    def next_cdc(self, stream: _SnowflakeCDCChangeStream) -> None:
        """Process the next CDC event.

        :param stream: _SnowflakeCDCChangeStream
        """
        logging.debug(f"[{self.identifier}] Waiting for next change in CDC stream")
        changes = stream.next()
        logging.debug(
            f"[{self.identifier}] Waiting for next change in CDC stream... DONE"
        )

        if changes:
            logging.info(f"[{self.identifier}] Received {len(changes)} changes")
            logging.debug(
                f"[{self.identifier}] Received ids: ",
                [r[self.primary_id] for r in changes],
            )

        # TODO group like changes together
        for change in changes:
            # action = change["METADATA$ACTION"]
            action = "INSERT"
            ids = [change[self.primary_id]]

            if action == "INSERT":
                event = DBEvent.insert
            else:
                event = DBEvent.delete

            self.event_handler(ids, event)
        time.sleep(1)

    def _drop_stream(self):
        logging.info(f"Drop stream {self.stream_name}...")
        if self.session:
            self.session.sql(f"DROP STREAM IF EXISTS {self.stream_name}").collect()

    def drop(self) -> None:
        """Drop the CDC stream and stop the listener."""
        self._drop_stream()

        self._stop_event.set()
        if self._scheduler:
            self._scheduler.join()

    def running(self) -> bool:
        """Check if the listener is running or not."""
        return not self._stop_event.is_set()
