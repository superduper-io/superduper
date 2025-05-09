import dataclasses as dc
import typing as t
import uuid
from abc import abstractmethod
from collections import defaultdict

import networkx as nx

from superduper import CFG, logging
from superduper.backends.base.backends import BaseBackend
from superduper.base.base import Base

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
    # components: t.List['CDC'] = _get_cdcs_on_table(table, db)
    context = str(uuid.uuid4())
    jobs: t.List['Job'] = []
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
        logging.info(f'Streaming with {component.component}:{component.identifier}')

    for job in jobs:
        job.execute(db)

    assert db.cluster is not None
    db.cluster.compute.release_futures(context)


def consume_events(events: t.List[Base], table: str, db: 'Datalayer'):
    """
    Consume events from table queue.

    :param events: List of events to be consumed.
    :param table: Queue Table.
    :param db: Datalayer instance.
    """
    if table != '_apply':
        logging.info(f'Consuming {len(events)} events on {table}.')
        consume_streaming_events(events=events, table=table, db=db)
    else:
        logging.info(f'Consuming {len(events)} _apply events')
        for event in events:
            event.execute(db)
        return
