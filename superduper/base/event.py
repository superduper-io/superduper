from abc import abstractmethod, ABC
import dataclasses as dc
import datetime
import typing as t
import uuid

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


@dc.dataclass(kw_only=True)
class Event(ABC):
    """Event dataclass to store event data.
    """

    def dict(self):
        """Convert to dict."""
        return {
            **dc.asdict(self),
            'genus': self.genus,
            'queue': self.queue,
        }

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
    type_id: t.ClassVar[str] = 'change'
    type: str
    queue: str
    ids: t.Sequence[str]

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
            for _, dep in component.dependencies:
                db.metadata.create_parent_child(component.uuid, dep)
        component.post_create(db=db)


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

    def __call__(self, db: 'Datalayer', futures: t.Dict, agent: t.Any | None = None):
        from superduper.backends.base.queue import Future
        if agent is None:
            agent = db.load(huuid=f'{self.type_id}:{self.identifier}:{self.uuid}')
        method = getattr(agent, self.method)
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
        return method(*args, **kwargs, dependencies=dependencies, context=self.context)

    def execute(self, db: 'Datalayer'):
        db.metadata.create_job({k: v for k, v in self.dict().items() if k not in {'genus', 'queue'}})
        return db.cluster.compute.submit(self)


events = {
    'signal': Signal,
    'change': Change,
    'create': Create,
    'job': Job,
}