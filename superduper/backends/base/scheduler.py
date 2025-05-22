import dataclasses as dc
import time
import typing as t
import uuid
from abc import abstractmethod
from collections import defaultdict

import networkx as nx

from superduper import CFG, logging
from superduper.backends.base.backends import BaseBackend
from superduper.base.base import Base
from superduper.base.event import Create, CreateTable, PutComponent, Update

DependencyType = t.Union[t.Dict[str, str], t.Sequence[t.Dict[str, str]]]

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer
    from superduper.base.metadata import Job

BATCH_SIZE = 100


class BaseScheduler(BaseBackend):
    """
    Base class for handling publisher and consumer process.

    This class is an implementation of message broker between
    producers (superduper db client) and consumers i.e listeners.
    """

    @abstractmethod
    def publish(self, events: t.List[Base]):
        """
        Publish events to local queue.

        :param events: list of events
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


def consume_streaming_events(events, table, db):
    """
    Consumer work from streaming events.

    Streaming event-types are {'insert', 'update', 'delete'}.

    :param events: list of events.
    :param table: table on which events were found.
    :param db: Datalayer instance.
    """
    out = defaultdict(lambda: [])
    for event in events:
        out[event.type].append(event)

    for event_type, events in out.items():
        ids: t.List[str] = sum([event.ids for event in events], [])
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
    context = str(uuid.uuid4())
    jobs: t.List['Job'] = []
    job_lookup: t.DefaultDict = defaultdict(dict)
    logging.info(f'Consuming {event_type} events on {table}')

    from superduper.components.cdc import build_streaming_graph

    G = build_streaming_graph(table, db)

    for huuid in nx.topological_sort(G):
        component = G.nodes[huuid]["component"]
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
        logging.info(f'Streaming with {component.component}:{component.identifier}')

    for job in jobs:
        job.execute(db)

    assert db.cluster is not None
    db.cluster.compute.release_futures(context)


def cluster_events(
    events: t.List[Base],
):
    """
    Cluster events into table, create and job events.

    :param events: List of events to be clustered.
    :return: Tuple of table events, create events and job events.
    """
    from superduper.base.metadata import Job

    table_events = []
    create_events = []
    job_events = []
    put_events = []
    for event in events:
        if isinstance(event, CreateTable):
            table_events.append(event)
        elif isinstance(event, (Update, Create)):
            create_events.append(event)
        elif isinstance(event, Job):
            job_events.append(event)
        elif isinstance(event, PutComponent):
            put_events.append(event)
    return table_events, create_events, put_events, job_events


def consume_events(
    events: t.List[Base],
    table: str,
    db: 'Datalayer',
    batch_size: int | None = None,
):
    """
    Consume events from table queue.

    :param events: List of events to be consumed.
    :param table: Queue Table.
    :param db: Datalayer instance.
    :param batch_size: Batch size for processing events.
    """
    if table != '_apply':
        logging.info(f'Consuming {len(events)} events on {table}.')
        consume_streaming_events(events=events, table=table, db=db)
    else:
        table_events, create_events, put_events, job_events = cluster_events(events)

        if table_events:
            start_time = time.time()
            logging.info(f'Consuming {len(table_events)} `CreateTable` events')
            CreateTable.batch_execute(
                events=table_events,
                db=db,
                batch_size=batch_size,
            )
            logging.info(
                f'Consumed {len(table_events)} `CreateTable` events in {time.time() - start_time:.2f}s'
            )

        if create_events:
            start_time = time.time()
            logging.info(f'Consuming {len(create_events)} `Create` events')
            Create.batch_execute(
                events=create_events,
                db=db,
                batch_size=batch_size,
            )
            logging.info(
                f'Consumed {len(create_events)} `Create` events in {time.time() - start_time:.2f}s'
            )

        if put_events:
            start_time = time.time()
            logging.info(f'Consuming {len(put_events)} `PutComponent` events')
            PutComponent.batch_execute(
                events=put_events,
                db=db,
                batch_size=batch_size,
            )
            logging.info(
                f'Consumed {len(put_events)} `PutComponent` events in {time.time() - start_time:.2f}s'
            )

        if job_events:
            start_time = time.time()
            logging.info(f'Consuming {len(job_events)} jobs (`Job`)')
            for job in job_events:
                job.execute(db)

            logging.info(
                f'Consumed {len(job_events)} jobs (`Job`) in {time.time() - start_time:.2f}s'
            )

        return
