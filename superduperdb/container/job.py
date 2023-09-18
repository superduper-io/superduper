import datetime
import typing as t
import uuid
from abc import abstractmethod

import superduperdb as s
from superduperdb.container.tasks import callable_job, method_job


def job(f):
    def wrapper(
        *args,
        distributed: t.Optional[bool] = None,
        db: t.Any = None,
        dependencies: t.Sequence[Job] = (),
        **kwargs,
    ):
        j = FunctionJob(callable=f, args=args, kwargs=kwargs)
        return j(db=db, distributed=distributed, dependencies=dependencies)

    return wrapper


class Job:
    """
    Base class for jobs. Jobs are used to run functions or methods on.

    :param args: positional arguments to be passed to the function or method
    :param kwargs: keyword arguments to be passed to the function or method
    :param identifier: unique identifier
    :param callable: function or method to be called
    :param db: DB instance to be used
    :param future: future object returned by dask
    """

    callable: t.Optional[t.Callable]

    def __init__(
        self,
        args: t.Optional[t.Sequence] = None,
        kwargs: t.Optional[t.Dict] = None,
    ):
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.identifier = str(uuid.uuid4())
        self.time = datetime.datetime.now()
        self.callable = None
        self.db = None
        self.future = None

    def watch(self):
        """
        Watch the stdout of the job.
        """
        return self.db.metadata.watch_job(identifier=self.identifier)

    def run_locally(self, db):
        """
        Run the job locally.
        """
        try:
            out = self.callable(*self.args, db=db, **self.kwargs)
            db.metadata.update_job(self.identifier, 'status', 'success')
        except Exception as e:
            db.metadata.update_job(self.identifier, 'status', 'failed')
            raise e
        return out

    @abstractmethod
    def run_on_dask(self, client, dependencies=()):
        """
        Run the job on a dask cluster.

        :param client: dask client
        :param dependencies: list of dependencies
        """
        raise NotImplementedError

    def dict(self):
        return {
            'identifier': self.identifier,
            'time': self.time,
            'status': 'pending',
            'args': self.args,
            'kwargs': self.kwargs,
            'stdout': [],
            'stderr': [],
        }

    def __call__(
        self, db: t.Any = None, distributed: t.Optional[bool] = None, dependencies=()
    ):
        """
        Run the job.

        :param db: DB instance to be used
        :param distributed: whether to run the job on a dask cluster
        :param dependencies: list of dependencies
        """
        raise NotImplementedError


class FunctionJob(Job):
    """
    Job for running a function.
    on a dask cluster.

    :param callable: function to be called
    :param args: positional arguments to be passed to the function
    :param kwargs: keyword arguments to be passed to the function
    """

    def __init__(
        self,
        callable: t.Callable,
        args: t.Optional[t.Sequence] = None,
        kwargs: t.Optional[t.Dict] = None,
    ):
        super().__init__(args=args, kwargs=kwargs)
        self.callable = callable

    def dict(self):
        d = super().dict()
        d['cls'] = 'FunctionJob'
        return d

    def run_on_dask(self, client, dependencies=()):
        """
        Run the job on a dask cluster.

        :param client: dask client
        :param dependencies: list of dependencies
        """
        CFG = s.configs.build_config()  # TODO Why not use s.CFG or s.CFG.deepcopy()
        self.future = client.submit(
            callable_job,
            cfg=CFG,
            function_to_call=self.callable,
            job_id=self.identifier,
            args=self.args,
            kwargs=self.kwargs,
            key=self.identifier,
            dependencies=dependencies,
        )
        return

    def __call__(
        self, db: t.Any = None, distributed: t.Optional[bool] = None, dependencies=()
    ):
        if db is None:
            from superduperdb.db.base.build import build_datalayer

            db = build_datalayer()

        if distributed is None:
            distributed = s.CFG.cluster.distributed
        self.db = db
        db.metadata.create_job(self.dict())
        if not distributed:
            self.run_locally(db)
        else:
            self.run_on_dask(client=db.distributed_client, dependencies=dependencies)
        return self


class ComponentJob(Job):
    """
    Job for running a class method of a component.

    :param component_identifier: unique identifier of the component
    :param type_id: type of the component
    :param method_name: name of the method to be called
    :param args: positional arguments to be passed to the method
    :param kwargs: keyword arguments to be passed to the method
    """

    def __init__(
        self,
        component_identifier: str,
        type_id: str,
        method_name: str,
        args: t.Optional[t.Sequence] = None,
        kwargs: t.Optional[t.Dict] = None,
    ):
        super().__init__(args=args, kwargs=kwargs)

        self.component_identifier = component_identifier
        self.method_name = method_name
        self.type_id = type_id
        self._component = None

    @property
    def component(self):
        return self._component

    @component.setter
    def component(self, value):
        self._component = value
        self.callable = getattr(self._component, self.method_name)

    def run_on_dask(self, client, dependencies=()):
        """
        Run the job on a dask cluster.

        :param client: dask client
        :param dependencies: list of dependencies
        """
        CFG = s.configs.build_config()  # Why?
        self.future = client.submit(
            method_job,
            cfg=CFG,
            type_id=self.type_id,
            identifier=self.component_identifier,
            method_name=self.method_name,
            job_id=self.identifier,
            args=self.args,
            kwargs=self.kwargs,
            key=self.identifier,
            dependencies=dependencies,
        )
        return

    def __call__(
        self, db: t.Any = None, distributed: t.Optional[bool] = None, dependencies=()
    ):
        if distributed is None:
            distributed = s.CFG.cluster.distributed
        if db is None:
            from superduperdb.db.base.build import build_datalayer

            db = build_datalayer()
        self.db = db
        db.metadata.create_job(self.dict())
        if self.component is None:
            self.component = db.load(self.type_id, self.component_identifier)
        if not distributed:
            self.run_locally(db)
        else:
            self.run_on_dask(client=db.distributed_client, dependencies=dependencies)
        return self

    def dict(self):
        d = super().dict()
        d.update(
            {
                'method_name': self.method_name,
                'component_identifier': self.component_identifier,
                'type_id': self.type_id,
                'cls': 'ComponentJob',
            }
        )
        return d
