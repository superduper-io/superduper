import dataclasses as dc
import datetime
import time
import typing as t
import uuid
from abc import ABC, abstractmethod

import numpy
import pandas

from superduper import logging
from superduper.backends.base.compute import ComputeBackend

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


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
        """Create event from kwargs.

        :param kwargs: kwargs to create event from
        """
        kwargs.pop('genus')
        kwargs.pop('queue')
        return cls(**kwargs)

    @abstractmethod
    def execute(
        self,
        db: 'Datalayer',
    ):
        """Execute the event.

        :param db: Datalayer instance
        """
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

    def execute(
        self,
        db: 'Datalayer',
    ):
        """Execute the signal.

        :param db: Datalayer instance.
        """
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
        """Create event from kwargs.

        :param kwargs: kwargs to create event from
        """
        kwargs.pop('genus')
        return cls(**kwargs)

    def execute(
        self,
        db: 'Datalayer',
    ):
        """Execute the change event.

        :param db: Datalayer instance.
        """
        raise NotImplementedError('Not relevant for this event class')


@dc.dataclass(kw_only=True)
class Create(Event):
    """
    Class for component creation events.

    :param context: the component context of creation.
    :param path: path of the component to be created
    :param data: the data of the component
    :param parent: the parent of the component (if any)
    :param children: the children of the component (if any)
    """

    genus: t.ClassVar[str] = 'create'
    queue: t.ClassVar[str] = '_apply'

    context: str
    path: str
    data: t.Dict
    parent: list | None = None
    children: t.List | None = None

    @property
    def component(self):
        return self.path.split('.')[-1]

    def execute(self, db: 'Datalayer'):
        """Execute the create event.

        :param db: Datalayer instance.
        """
        # TODO decide where to assign version
        logging.info(
            f'Creating {self.path.split("/")[-1]}:'
            f'{self.data["identifier"]}:{self.data["uuid"]}'
        )

        artifact_ids, _ = db._find_artifacts(self.data)

        db.metadata.create_artifact_relation(
            component=self.component,
            identifier=self.data['identifier'],
            uuid=self.data['uuid'],
            artifact_ids=artifact_ids,
        )

        db.metadata.create_component(self.data, path=self.path)
        component = db.load(component=self.component, uuid=self.data['uuid'])

        if self.children:
            for child in self.children:
                db.metadata.create_parent_child(
                    child_component=child[0],
                    child_identifier=child[1],
                    child_uuid=child[2],
                    parent_component=self.component,
                    parent_identifier=component.identifier,
                    parent_uuid=component.uuid,
                )

        component.on_create()

    @property
    def huuid(self):
        """Return the hashed uuid."""
        return f'{self.component}:' f'{self.data["identifier"]}:' f'{self.data["uuid"]}'


@dc.dataclass(kw_only=True)
class Delete(Event):
    """
    Class for component deletion events.

    :param component: the type of component to be created
    :param identifier: the identifier of the component to be deleted
    """

    genus: t.ClassVar[str] = 'delete'
    queue: t.ClassVar[str] = '_apply'

    component: str
    identifier: str

    @property
    def huuid(self):
        """Return the hashed uuid."""
        return f'{self.component}:{self.identifier}'

    def execute(self, db: 'Datalayer'):
        """Execute the delete event.

        :param db: Datalayer instance.
        """
        object = db.load(component=self.component, identifier=self.identifier)

        db.metadata.delete_component(self.component, self.identifier)
        artifact_ids = db.metadata.get_artifact_relations_for_component(
            self.component, self.identifier
        )

        if artifact_ids:
            parents_to_artifacts = db.metadata.get_artifact_relations_for_artifacts(
                artifact_ids
            )
            df = pandas.DataFrame(parents_to_artifacts)
            if not df.empty:
                condition = numpy.logical_or(
                    df['component'] != self.component,
                    df['identifier'] != self.identifier,
                )
                other_relations = df[condition]
                to_exclude = other_relations['artifact_id'].tolist()
                artifact_ids = sorted(list(set(artifact_ids) - set(to_exclude)))
            db.artifact_store.delete_artifact(artifact_ids)

        db.metadata.delete_parent_child_relationships(
            parent_component=self.component,
            parent_identifier=self.identifier,
        )

        object.cleanup()


@dc.dataclass(kw_only=True)
class Update(Event):
    """
    Update component event.

    :param context: the component context of creation.
    :param component: the type of component to be created
    :param data: the component data to be created
    :param parent: the parent of the component (if any)
    """

    genus: t.ClassVar[str] = 'update'
    queue: t.ClassVar[str] = '_apply'

    context: str
    component: str
    data: t.Dict
    parent: list | None = None

    def execute(
        self,
        db: 'Datalayer',
    ):
        """Execute the create event.

        :param db: Datalayer instance.
        """
        artifact_ids, _ = db._find_artifacts(self.data)
        db.metadata.create_artifact_relation(
            component=self.component,
            identifier=self.data['identifier'],
            uuid=self.data['uuid'],
            artifact_ids=artifact_ids,
        )
        db.metadata.replace_object(
            self.component, uuid=self.data['uuid'], info=self.data
        )

    @property
    def huuid(self):
        """Return the hashed uuid."""
        return f'{self.component}' f'{self.data["identifier"]}:' f'{self.data["uuid"]}'


@dc.dataclass(kw_only=True)
class Job(Event):
    """
    Job event.

    :param context: context component for job creation
    :param component: type of component
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
    component: str
    identifier: str
    uuid: str
    args: t.Sequence[t.Any] = ()
    kwargs: t.Dict = dc.field(default_factory=dict)
    time: datetime.datetime = dc.field(default_factory=datetime.datetime.now)
    job_id: t.Optional[str] = dc.field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    status: str = 'pending'
    dependencies: t.List[str] = dc.field(default_factory=list)

    def get_status(self, db):
        """Get the status of the job.

        :param db: Datalayer instance
        """
        return db['Job'].get(job_id=self.job_id)['status']

    def wait(self, db: 'Datalayer', heartbeat: float = 1, timeout: int = 60):
        """Wait for the job to finish.

        :param db: Datalayer instance
        :param heartbeat: time to wait between checks
        :param timeout: timeout in seconds
        """
        start = time.time()
        status = 'pending'
        while (status := self.get_status(db)) in {
            'pending',
            'running',
        } and time.time() - start < timeout:
            if status == 'pending':
                logging.info(f'Job {self.job_id} is pending')
            elif status == 'running':
                logging.info(f'Job {self.job_id} is running')
            else:
                break

            time.sleep(heartbeat)

        logging.info(f'Job {self.job_id} finished with status: {status}')
        return status

    @property
    def huuid(self):
        """Return the hashed uuid."""
        return f'{self.component}:{self.identifier}:{self.uuid}.{self.method}'

    def get_args_kwargs(self, futures):
        """Get args and kwargs for job execution.

        :param futures: dict of futures
        """
        from superduper.backends.base.scheduler import Future

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

    def execute(
        self,
        db: 'Datalayer',
    ):
        """Execute the job event.

        :param db: Datalayer instance
        """
        meta = {k: v for k, v in self.dict().items() if k not in {'genus', 'queue'}}
        db.metadata.create_job(meta)
        return db.cluster.compute.submit(self)


events = {
    'signal': Signal,
    'change': Change,
    'update': Update,
    'create': Create,
    'job': Job,
}


def unpack_event(dict):
    """
    Helper function to deserialize event into Event class.

    :param dict: Serialized event.
    """
    event_type = events[dict.get('genus')]
    return event_type.create(dict)
