import dataclasses as dc
import networkx as nx
import typing as t
from abc import ABC, abstractmethod
from collections import defaultdict
import uuid

from superduper import logging, CFG
from superduper.backends.base.backends import BaseBackend
from superduper.base.event import Event

DependencyType = t.Union[t.Dict[str, str], t.Sequence[t.Dict[str, str]]]

if t.TYPE_CHECKING:
    from superduper.components.cdc import CDC
    from superduper.base.datalayer import Datalayer


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
        self.futures = defaultdict(lambda: {})

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


class JobFutureException(Exception):
    """Exception when futures are not ready.
    
    # noqa
    """
    ...


# def _get_cdcs_on_table(table, db: 'Datalayer'):
#     cdcs = db.metadata.show_cdcs(table)
#     out = []
#     for uuid in cdcs:
#         out.append(
#             db.load(uuid=uuid)
#         )
#     return out


def _get_cdcs_on_table(table, db: 'Datalayer'):
    cdcs = db.metadata.show_cdcs(table)
    from superduper.components.listener import Listener
    out = []
    for uuid in cdcs:
        component = db.load(uuid=uuid)
        if isinstance(component, Listener):
            if len(component.select.tables) > 1:
                continue
            out.append(component)
        out.append(component)
    return out


def _get_parent_cdcs_of_component(component, db: 'Datalayer'):
    parents = db.metadata.get_component_version_parents(component.uuid)
    out = []
    for uuid in parents:
        r = db.metadata.get_component_by_uuid(uuid)
        if r.get('cdc_table'):
            out.append(db.load(uuid=uuid))
    return {c.uuid: c for c in out}


def _get_parent_cdcs_of_components(components, db):
    out = {}
    for component in components:
        out.update(
            _get_parent_cdcs_of_component(component, db=db)
        )
    return list(out.values())


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
    Future output

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
    jobs = []
    job_lookup = defaultdict(dict)

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
        print(f'Streaming with {component.type_id}:{component.identifier}')

    for job in jobs:
        db.cluster.compute.submit(job)
    db.cluster.compute.release_futures(context)
