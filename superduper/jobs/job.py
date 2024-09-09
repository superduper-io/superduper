import dataclasses as dc
import datetime
import typing as t
import uuid


if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


@dc.dataclass
class Job:
    """
    Base class for jobs. Jobs are used to run functions or methods on.

    :param type_id: Component type_id
    :param identifier: Component identifier
    :param uuid: Component uuid
    :param method: Method name of Component
    :param job_id: Job identifier
    :param args: positional arguments to be passed to the method
    :param kwargs: keyword arguments to be passed to the method
    :param db: A datalayer instance
    :param time: Time of job creation
    """
    type_id: str
    identifier: str
    uuid: str
    method: str
    job_id: t.Optional[str] = dc.field(default_factory=lambda: str(uuid.uuid4()))
    args: t.Sequence = dc.field(default_factory=tuple)
    kwargs: t.Dict = dc.field(default_factory=dict)
    db: dc.InitVar['Datalayer'] = None
    time: datetime.datetime = dc.field(default_factory=lambda: str(datetime.datetime.now()))

    def __post_init__(self, db):
        self.db: 'Datalayer' = db

    def submit(self, dependencies=()):
        """Submit job for execution.

        :param dependencies: list of dependencies
        """
        return self.db.compute.submit(self, dependencies=dependencies)

    def dict(self):
        """Return a dictionary representation of the job."""
        return dc.asdict(self)

    def __call__(self, dependencies=()):
        """Run the job."""
        self.db.metadata.create_job(self.dict())
        return self.submit(dependencies=dependencies)