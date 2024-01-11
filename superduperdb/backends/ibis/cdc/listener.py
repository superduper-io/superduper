import threading
import time
import typing as t

from superduperdb import CFG, logging
from superduperdb.backends.ibis import query
from superduperdb.backends.ibis.cdc.base import IbisDBPacket
from superduperdb.base.config import LogBasedStrategy, PollingStrategy
from superduperdb.cdc import cdc
from superduperdb.misc.runnable.runnable import Event

if t.TYPE_CHECKING:
    from superduperdb.backends.ibis.query import Table
    from superduperdb.base.datalayer import Datalayer


class PollingStrategyIbis:
    def __init__(
        self, db: 'Datalayer', table: 'Table', strategy, primary_id: str = 'id'
    ):
        self.db = db
        self.table = table
        self.strategy = strategy

        self.primary_id = primary_id
        self.increment_field = strategy.auto_increment_field
        self.frequency = strategy.frequency
        self._last_processed_id = -1

    def fetch_ids(self):
        raise NotImplementedError

    def post_handling(self):
        time.sleep(self.frequency)

    def get_strategy(self):
        if self.increment_field:
            return PollingStrategyIbisByIncrement(
                self.db, self.table, self.strategy, primary_id=self.primary_id
            )
        else:
            return PollingStrategyIbisByID(
                self.db, self.table, self.strategy, primary_id=self.primary_id
            )


class PollingStrategyIbisByIncrement(PollingStrategyIbis):
    def fetch_ids(
        self,
    ):
        assert self.increment_field
        _filter = self.table.__getattr__(self.increment_field) > self._last_processed_id
        query = self.table.select(self.primary_id).filter(_filter)
        ids = list(self.db.execute(query))
        ids = [id[self.primary_id] for id in ids]
        self._last_processed_id = int(max(ids)) if ids else self._last_processed_id
        return ids


class PollingStrategyIbisByID(PollingStrategyIbis):
    ...


class IbisDatabaseListener(cdc.BaseDatabaseListener):
    """
    It is a class which helps capture data from ibis database and handle it
    accordingly.

    This class accepts options and db instance from user and starts a scheduler
    which could schedule a listening service to listen change stream.

    This class builds a workflow graph on each change observed.

    """

    DEFAULT_ID: str = 'id'
    EXCLUSION_KEYS: t.Sequence[str] = [DEFAULT_ID]
    IDENTITY_SEP: str = '/'
    _scheduler: t.Optional[threading.Thread]

    def __init__(
        self,
        db: 'Datalayer',
        on: query.Table,
        stop_event: Event,
        identifier: 'str' = '',
        timeout: t.Optional[float] = None,
        strategy: t.Optional[t.Union['PollingStrategy', 'LogBasedStrategy']] = None,
    ):
        """__init__.

        :param db: It is a superduperdb instance.
        :param on: It is used to define a Collection on which CDC would be performed.
        :param stop_event: A threading event flag to notify for stoppage.
        :param identifier: A identifier to represent the listener service.
        :param strategy: Used to select strategy used for listening changes
                        options:
                            PollingStrategy (This strategy polls table every
                                            `frequency` seconds, more info at
                                            superduperdb.cdc.cdc.PollingStrategy)
                            LogBasedStrategy (Not implemented yet)
        """

        if not strategy:
            assert CFG.cluster.cdc
            self.strategy = CFG.cluster.cdc.strategy
        else:
            self.strategy = strategy

        self.db_type = 'ibis'
        self.packet = lambda ids, query, event_type: IbisDBPacket(
            ids, query, event_type
        )

        super().__init__(
            db=db, on=on, stop_event=stop_event, identifier=identifier, timeout=timeout
        )

    def on_update(self, ids: t.Sequence, db: 'Datalayer', table: query.Table) -> None:
        raise NotImplementedError

    def on_delete(self, ids: t.Sequence, db: 'Datalayer', table: query.Table) -> None:
        raise NotImplementedError

    def on_create(self, ids: t.Sequence, db: 'Datalayer', table: query.Table) -> None:
        """on_create.
        A helper on create event handler which handles inserted document in the
        change stream.
        It basically extracts the change document and build the taskflow graph to
        execute.

        :param ids: Changed row ids.
        :param db: a superduperdb instance.
        :param table: The table on which change was observed.
        """
        logging.debug('Triggered `on_create` handler.')
        self.create_event(
            ids=ids, db=db, table_or_collection=table, event=cdc.DBEvent.insert
        )

    def setup_cdc(self):
        """
        Setup cdc change stream from user provided
        """
        if isinstance(self.strategy, PollingStrategy):
            self.stream = PollingStrategyIbis(
                self.db,
                self._on_component,
                strategy=self.strategy,
                primary_id=self.DEFAULT_ID,
            ).get_strategy()
        elif isinstance(self.strategy, cdc.LogBasedStrategy):
            raise NotImplementedError('logbased strategy not implemented yet')
        else:
            raise TypeError(f'{self.strategy} is not a valid strategy')
        return self.stream

    def next_cdc(self, stream) -> None:
        """
        Get the next stream of change observed on the given `Collection`.
        """
        ids = stream.fetch_ids()
        if ids:
            # Harcoded with insert since delete and update not supported
            self.event_handler(ids, event=cdc.DBEvent.insert)
        stream.post_handling()

    def listen(
        self,
    ) -> None:
        try:
            self._stop_event.clear()
            if self._scheduler:
                if self._scheduler.is_alive():
                    raise RuntimeError(
                        'CDC Listener thread is already running!,/'
                        'Please stop the listener first.'
                    )

            self._scheduler = cdc.DatabaseListenerThreadScheduler(
                self, stop_event=self._stop_event, start_event=self._startup_event
            )

            assert self._scheduler is not None
            self._scheduler.start()

            self._startup_event.wait(timeout=self.timeout)
        except Exception:
            logging.error('Listening service stopped!')
            self.stop()
            raise

    def stop(self) -> None:
        """
        Stop listening cdc changes.
        This stops the corresponding services as well.
        """
        self._stop_event.set()
        if self._scheduler:
            self._scheduler.join()

    def running(self) -> bool:
        return not self._stop_event.is_set()
