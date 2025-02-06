import dataclasses as dc
import typing as t
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict

import networkx as nx

from superduper import CFG, logging
from superduper.backends.base.backends import BaseBackend
from superduper.base.event import Event, Job

DependencyType = t.Union[t.Dict[str, str], t.Sequence[t.Dict[str, str]]]

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer

BATCH_SIZE = 100


def _chunked_list(lst, batch_size=BATCH_SIZE):
    if len(lst) <= batch_size:
        return [lst]

    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


class BaseQueueConsumer(ABC):
    """
    Base class for handling consumer process.

    This class is an implementation of message broker between
    producers (superduper db client) and consumers i.e listeners.

    :param uri: Uri to connect.
    :param queue_name: Queue to consume.
    :param callback: Callback for consumed messages.
    """

    def __init__(
        self,
        uri: t.Optional[str] = '',
        queue_name: str = '',
        callback: t.Optional[t.Callable] = None,
    ):
        self.uri = uri
        self.callback = callback
        self.queue_name = queue_name
        self.futures: t.DefaultDict = defaultdict(lambda: {})

    @abstractmethod
    def start_consuming(self):
        """Abstract method to start consuming messages."""
        pass

    @abstractmethod
    def close_connection(self):
        """Abstract method to close connection."""
        pass

    def consume(self, *args, **kwargs):
        """Start consuming messages from queue."""
        logging.info(f"Started consuming on queue: {self.queue_name}")
        try:
            self.start_consuming()
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt: Stopping consumer...")
        finally:
            self.close_connection()
            logging.info(f"Stopped consuming on queue: {self.queue_name}")


class BaseQueuePublisher(BaseBackend):
    """
    Base class for handling publisher and consumer process.

    This class is an implementation of message broker between
    producers (superduper db client) and consumers i.e listeners.

    :param uri: Uri to connect.
    """

    def __init__(self, uri: t.Optional[str]):
        super().__init__()
        self.uri: t.Optional[str] = uri
        self.queue: t.Dict = defaultdict(lambda: [])

    def clear(self):
        """Clear the queue."""
        self.queue = defaultdict(lambda: [])

    @abstractmethod
    def build_consumer(self, **kwargs):
        """Build a consumer instance."""

    @abstractmethod
    def publish(self, events: t.List[Event]):
        """
        Publish events to local queue.

        :param events: list of events
        :param to: Component name for events to be published.
        """

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


class JobFutureException(Exception):
    """Exception when futures are not ready.

    # noqa
    """

    ...


def consume_streaming_events(events, table, db):
    """
    Consumer work from streaming events.

    Streaming event-types are {'insert', 'update', 'delete'}.

    :param events: list of events
    :param table: table on which events were found
    :param db: Datalayer instance
    """
    out = defaultdict(lambda: [])
    for event in events:
        out[event.type].append(event)

    for event_type, events in out.items():
        ids = sum([event.ids for event in events], [])
        _consume_event_type(
            event_type,
            ids=ids,
            table=table,
            db=db,
        )


@dc.dataclass
class Future:
    """
    Future output.

    :param job_id: job identifier
    """

    job_id: str


def _consume_event_type(event_type, ids, table, db: 'Datalayer'):
    # contains all components triggered by the table
    # and all components triggered by the output of these components etc.
    # "uuid" -> dict("trigger_method": future)
    logging.debug(table)
    # components: t.List['CDC'] = _get_cdcs_on_table(table, db)
    context = str(uuid.uuid4())
    jobs: t.List[Job] = []
    job_lookup: t.DefaultDict = defaultdict(dict)
    logging.info(f'Consuming {event_type} events on {table}')

    from superduper.components.cdc import build_streaming_graph

    G, components = build_streaming_graph(table, db)

    for huuid in nx.topological_sort(G):
        component = components[huuid]
        # this is a dictionary/ mapping method_name -> future
        # try this until the dependencies are there
        input_table = component.cdc_table
        if input_table.startswith(CFG.output_prefix):
            input_uuid = input_table.split('__')[-1]
            # Maybe think about generalizing this
            # This is getting the "run" method from `Listener`
            input_ids = Future(job_lookup[input_uuid]['run'])
        else:
            input_ids = ids

        # For example for "Listener" this will create
        # the Listener.run job only
        sub_jobs = component.create_jobs(
            context=context,
            ids=input_ids,
            jobs=jobs,
            event_type=event_type,
        )

        for job in sub_jobs:
            job_lookup[component.uuid][job.method] = job.job_id
        jobs += sub_jobs
        logging.info(f'Streaming with {component.type_id}:{component.identifier}')

    if db.metadata.batched:
        for chunk in _chunked_list(jobs):
            for job in chunk:
                job.execute(db)
            db.metadata.commit()
    else:
        for job in jobs:
            job.execute(db)

    db.cluster.compute.release_futures(context)


table_type_ids = {'table', 'schema', 'data', 'datatype', 'dataset'}


def consume_events(events, table: str, db=None):
    """
    Consume events from table queue.

    :param events: List of events to be consumed.
    :param table: Queue Table.
    :param db: Datalayer instance.
    """
    if table != '_apply':
        consume_streaming_events(events=events, table=table, db=db)
    else:
        if not db.metadata.batched:
            for event in events:
                event.execute(db)
            return

        # table events
        table_events = []
        non_table_events = []

        for ix, event in enumerate(events):
            if event.genus == 'create' and event.component['type_id'] in table_type_ids:
                table_events.append(event)
            else:
                non_table_events.append(event)

        for event in table_events:
            event.execute(db)

        db.metadata.commit()

        # non table events
        for event in non_table_events:
            event.execute(db)
            if event.genus == 'update':
                db.metadata.commit()

        db.metadata.commit()
