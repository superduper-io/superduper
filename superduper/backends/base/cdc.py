import threading
import typing as t
from abc import ABC, abstractmethod
from enum import Enum

from superduper import logging
from superduper.backends.base.backends import BaseBackend
from superduper.misc.threading import Event

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


class DBEvent(str, Enum):
    """Enum to hold database basic events.

    Represents the types of events that can occur in a database.

    # noqa
    """

    delete = "delete"
    insert = "insert"
    update = "update"
    upsert = "upsert"


class BaseDatabaseListener(ABC):
    """Base class which defines basic functions for database listeners.

    This class is responsible for defining the basic functions
    that need to be implemented by the database listener.

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
        self.db = db
        self.table = table

        self._stop_event = stop_event
        self._startup_event = Event()
        self._scheduler = None
        self.timeout = timeout
        self.error_handler = error_handler

    @property
    def identifier(self):
        """Get the identifier for this listener.

        :return: Identifier (table name)
        """
        return self.table

    def initialize(self):
        """Listen to database changes.

        Sets up and starts the database listener thread.

        :return: Self for method chaining
        :raises RuntimeError: If listener thread is already running
        :raises Exception: If any error occurs during initialization
        """
        try:
            self._stop_event.clear()
            if self._scheduler:
                if self._scheduler.is_alive():
                    raise RuntimeError(
                        "CDC Listener thread is already running!,/"
                        "Please stop the listener first."
                    )

            logging.info("Build CDC Listener service")
            self._scheduler = DatabaseListenerThreadScheduler(
                self, stop_event=self._stop_event, start_event=self._startup_event
            )
            logging.info("Build CDC Listener service... DONE")

            assert self._scheduler is not None

            logging.info("Starting CDC Listener service")
            self._scheduler.start()
            logging.info("Starting CDC Listener service... DONE")

            self._startup_event.wait(timeout=self.timeout)
        except Exception as e:
            logging.error("Listening service stopped!")
            self.drop()
            raise e
        return self

    def drop(self):
        """Stop listening to CDC changes.

        This stops the corresponding services as well.
        """
        self._stop_event.set()
        if self._scheduler and self._scheduler.is_alive():
            try:
                self._scheduler.join(timeout=5)
            except Exception as e:
                logging.error(f"Error while stopping the listener: {e}")

    def running(self) -> bool:
        """Check if the listener is running or not.

        :return: True if the listener is running, False otherwise
        """
        return not self._stop_event.is_set()

    @abstractmethod
    def setup_cdc(self):
        """Set up the CDC listener.

        :return: CDC stream object
        """
        pass

    @abstractmethod
    def next_cdc(self, stream) -> None:
        """Process the next CDC event.

        :param stream: CollectionChangeStream
        """
        pass

    def create_event(
        self,
        ids: t.List[str],
        db: "Datalayer",
        table: str,
        event: DBEvent,
    ):
        """Create an event.

        A helper to create packet based on the event type and put it on the CDC queue.

        :param ids: Document ids
        :param db: A superduper instance
        :param table: The collection on which change was observed
        :param event: CDC event type
        """
        table = table
        table = table if isinstance(table, str) else table.table
        logging.info(f"Detected CDC Event: {event} on {table} with ids: {ids}")
        db.on_event(table=table, ids=ids, event_type=event)
        logging.info('Event created and put on the queue')

    def event_handler(self, ids: t.List[str], event: DBEvent) -> None:
        """Handle the incoming change stream event.

        A helper function to handle incoming changes from change stream on a collection.

        :param ids: Changed document ids
        :param event: CDC event
        """
        self.create_event(ids=ids, db=self.db, table=self.table, event=event)


class DatabaseListenerThreadScheduler(threading.Thread):
    """Thread scheduler to listen to CDC changes.

    This class is responsible for listening to the CDC changes and
    executing the following job.

    :param listener: A BaseDatabaseListener instance
    :param stop_event: A threading event flag to notify for stoppage
    :param start_event: A threading event flag to notify for start
    """

    def __init__(
        self,
        listener: BaseDatabaseListener,
        stop_event: Event,
        start_event: Event,
    ) -> None:
        threading.Thread.__init__(self, daemon=True)
        self.stop_event = stop_event
        self.start_event = start_event
        self.listener = listener

    def run(self) -> None:
        """Start to listen to the CDC changes.

        Sets up CDC stream and processes events until stopped.
        """

        def setup():
            """Set up the CDC stream.

            :return: CDC stream object
            """
            cdc_stream = self.listener.setup_cdc()
            self.start_event.set()
            return cdc_stream

        while True:
            try:
                cdc_stream = setup()
                break
            except Exception as e:
                if self.listener.error_handler:
                    self.listener.error_handler(e)

        while not self.stop_event.is_set():
            try:
                self.listener.next_cdc(cdc_stream)
            except Exception as e:
                if self.listener.error_handler:
                    self.listener.error_handler(e, self.listener)


class CDCBackend(BaseBackend):
    """Base backend for CDC."""

    @abstractmethod
    def handle_event(self, event_type, table, ids):
        """Handle an incoming event.

        :param event_type: The type of event.
        :param table: The table to handle.
        :param ids: The ids to handle.
        """
        pass

    @property
    def db(self) -> 'Datalayer':
        """Get the ``db``."""
        return self._db

    @db.setter
    def db(self, value: 'Datalayer'):
        """Set the ``db``.

        :param value: ``Datalayer`` instance.
        """
        self._db = value
