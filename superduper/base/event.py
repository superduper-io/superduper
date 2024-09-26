import dataclasses as dc
import datetime
import typing as t
import uuid
from abc import ABC, abstractmethod

import networkx as nx
from rich.console import Console
from rich.text import Text
from rich.tree import Tree

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


@dc.dataclass
class DummyJob:
    """
    Dummy job dataclass to store job data.

    # noqa
    """

    job_id: str
    dependencies: t.List[str] = dc.field(default_factory=list)
    method: str = 'dummy'

    @property
    def huuid(self):
        return f'{self.job_id}'


@dc.dataclass(kw_only=True)
class DeploymentPlan:
    """
    Deployment plan dataclass to store deployment plan data.

    :param create_events: list of create events
    :param jobs: list of job events

    >>> job1 = DummyJob(job_id='job1')
    >>> job2 = DummyJob(job_id='job2', dependencies=['job1'])
    >>> plan = DeploymentPlan(create_events=[], jobs=[job1, job2])
    >>> plan.show(False)
    DEPLOYMENT PLAN
    ├── CREATE
    └── JOBS
        └── msg-job2: dummy
            └── msg-job1: dummy
    """

    create_events: t.List['Create']
    jobs: t.List['Job']

    @staticmethod
    def add_nodes_to_tree(tree, node, G, lookup, style=True):
        node_label = Text()

        node_label.append(lookup[node].huuid, style='bold magenta' if style else None)
        node_label.append(": ", style='bold white' if style else None)
        node_label.append(lookup[node].method, style='dim' if style else None)

        subtree = tree.add(node_label)

        for child in G.successors(node):
            DeploymentPlan.add_nodes_to_tree(subtree, child, G, lookup, style=style)

    def show(self, style=True):
        create_tree = Tree('CREATE')

        lookup = {job.job_id: job for job in self.jobs}

        G = nx.DiGraph()
        for job in self.jobs:
            for dep in job.dependencies:
                G.add_edge(job.job_id, dep)

        root = next(n for n, d in G.in_degree() if d == 0)
        if style:
            job_tree = Tree("JOBS", guide_style='bold bright_blue')
        else:
            job_tree = Tree("JOBS")

        assert all([job.job_id in lookup for job in self.jobs])
        self.add_nodes_to_tree(job_tree, root, G, lookup, style=style)

        merged_tree = Tree('DEPLOYMENT PLAN')
        merged_tree.add(create_tree)
        merged_tree.add(job_tree)

        console = Console()
        console.print(merged_tree)


@dc.dataclass(kw_only=True)
class Event(ABC):
    """Event dataclass to store event data."""

    def dict(self):
        """Convert to dict."""
        _base_dict = dc.asdict(self)
        if 'time' in _base_dict:
            _base_dict['time'] = str(_base_dict['time'])
        return {
            **_base_dict,
            'genus': self.genus,
            'queue': self.queue,
        }

    @classmethod
    def create(cls, kwargs):
        kwargs.pop('genus')
        kwargs.pop('queue')
        return cls(**kwargs)

    @abstractmethod
    def execute(self, db: 'Datalayer'):
        pass


@dc.dataclass(kw_only=True)
class Signal(Event):
    """
    Event used to send a signal to the scheduler.

    :param msg: signal to send
    :param context: the context of component creation
    """

    genus: t.ClassVar[str] = 'signal'
    queue: t.ClassVar[str] = '_apply'
    msg: str
    context: str

    def execute(self, db: 'Datalayer'):
        if self.msg.lower() == 'done':
            db.cluster.compute.release_futures(self.context)


@dc.dataclass(kw_only=True)
class Change(Event):
    """
    Class for streaming change events.

    :param ids: list of ids detected in databackend.
    :param type: {'insert', 'update', 'delete'}
    :param queue: which table was affected
    :param ids: the ids affected
    """

    genus: t.ClassVar[str] = 'change'
    type: str
    queue: str
    ids: t.Sequence[str]

    @classmethod
    def create(cls, kwargs):
        kwargs.pop('genus')
        return cls(**kwargs)

    def execute(self, db: 'Datalayer'):
        raise NotImplementedError('Not relevant for this event class')


@dc.dataclass(kw_only=True)
class Create(Event):
    """
    Class for component creation events.

    :param context: the component context of creation.
    :param component: the component to be created
    :param parent: the parent of the component (if any)
    """

    genus: t.ClassVar[str] = 'create'
    queue: t.ClassVar[str] = '_apply'

    context: str
    component: t.Dict
    parent: str | None = None

    def execute(self, db: 'Datalayer'):
        # TODO decide where to assign version
        db.metadata.create_component(self.component)
        component = db.load(uuid=self.component['uuid'])
        if self.parent:
            db.metadata.create_parent_child(self.parent, component.uuid)

        if hasattr(component, 'dependencies') and component.dependencies:
            for dep in component.dependencies:
                db.metadata.create_parent_child(component.uuid, dep)
        component.on_create(db=db)

    @property
    def huuid(self):
        return f'{self.component["type_id"]}:{self.component["identifier"]}:{self.component["uuid"]}'


@dc.dataclass(kw_only=True)
class Job(Event):
    """
    Job event.

    :param context: context component for job creation
    :param type_id: type_id of component
    :param identifier: identifier of component
    :param uuid: uuid of component
    :param args: arguments of method
    :param kwargs: kwargs of method
    :param time: time of job creation
    :param job_id: id of job
    :param method: method to run
    :param status: status of job
    :param dependencies: list of job_id dependencies
    """

    genus: t.ClassVar[str] = 'job'
    queue: t.ClassVar[str] = '_apply'

    context: str
    type_id: str
    identifier: str
    uuid: str
    args: t.Sequence[t.Any] = ()
    kwargs: t.Dict = dc.field(default_factory=dict)
    time: datetime.datetime = dc.field(default_factory=datetime.datetime.now)
    job_id: t.Optional[str] = dc.field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    status: str = 'pending'
    dependencies: t.Sequence[str] = ()

    @property
    def huuid(self):
        return f'{self.type_id}:{self.identifier}:{self.uuid}'

    def get_args_kwargs(self, futures):
        from superduper.backends.base.queue import Future

        dependencies = []
        if self.dependencies:
            dependencies = [futures[k] for k in self.dependencies if k in futures]
        args = []
        for arg in self.args:
            if isinstance(arg, Future):
                args.append(futures[arg.job_id])
            else:
                args.append(arg)
        kwargs = {}
        for k, v in self.kwargs.items():
            if isinstance(v, Future):
                kwargs[k] = futures[v.job_id]
            else:
                kwargs[k] = v
        kwargs['dependencies'] = dependencies
        return args, kwargs

    def execute(self, db: 'Datalayer'):
        db.metadata.create_job(
            {k: v for k, v in self.dict().items() if k not in {'genus', 'queue'}}
        )
        return db.cluster.compute.submit(self)


events = {
    'signal': Signal,
    'change': Change,
    'create': Create,
    'job': Job,
}


def unpack_event(dict):
    """
    Helper function to deserialize event
    into Event class.

    :param dict: Serialized event.
    """
    event_type = events[dict.get('genus')]
    return event_type.create(dict)
