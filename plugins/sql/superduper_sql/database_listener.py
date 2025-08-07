import threading
import time
import typing as t

from superduper import logging
from superduper.backends.base.cdc import BaseDatabaseListener, DBEvent
from superduper.misc.threading import Event

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


class PollingStream:
    """A simple polling stream to fetch changes from a SQL database.

    :param db: A superduper instance
    :param table: A table or collection on which the listener is invoked
    """

    def __init__(self, db: "Datalayer", table: str):
        self.db = db
        self.table = table
        self._processed_ids = db[table].ids()

    def fetch_ids(
        self,
    ):
        """Fetch the IDs of the changes in the table since the last check."""
        ids = self.db[self.table].ids()
        change_ids = set(ids) - set(self._processed_ids)
        self._processed_ids = list(set(self._processed_ids + list(change_ids)))
        if change_ids:
            logging.info(f"Found {len(change_ids)} changes in {self.table}")
            logging.debug("-" * 50)
            logging.debug(f"Fetched IDS: {self.table}: {change_ids}")
        return change_ids


class SQLDatabaseListener(BaseDatabaseListener):
    """SQL specific database listener implementation.

    :param db: A superduper instance
    :param table: A table or collection on which the listener is invoked
    :param stop_event: A threading event flag to notify for stoppage
    :param timeout: A timeout for the listener
    :param error_handler: A callable to handle errors during listening
    """

    DEFAULT_ID: str = "id"
    IDENTITY_SEP: str = "/"
    _scheduler: t.Optional[threading.Thread]

    def __init__(
        self,
        db: "Datalayer",
        table: str,
        stop_event: Event,
        timeout: t.Optional[float] = None,
        error_handler: t.Optional[t.Callable] = None,
    ):
        super().__init__(
            db=db,
            table=table,
            stop_event=stop_event,
            timeout=timeout,
            error_handler=error_handler,
        )
        self.primary_id = db[table].primary_id.execute()

        try:
            frequency = int(db.cfg.cluster.cdc.strategy["frequency"])
            logging.info(
                f"CDC frequency set in the config: {table}: {frequency} seconds"
            )
        except Exception:
            frequency = 30
            logging.warning(
                "CDC frequency not set in the config, using default value of 30 seconds"
            )
        self.frequency = frequency

    def setup_cdc(self):
        """Set up the CDC listener for SQL databases."""
        self.stream = PollingStream(
            self.db,
            self.table,
        )
        return self.stream

    def next_cdc(self, stream) -> None:
        """Process the next CDC event."""
        ids = stream.fetch_ids()
        if ids:
            logging.info(f"Found a change using CDC: {self.table}: {ids}")
            assert len(ids) == len(set(ids))
            # Harcoded with insert since delete and update not supported
            self.event_handler(ids, event=DBEvent.insert)

        time.sleep(self.frequency)
