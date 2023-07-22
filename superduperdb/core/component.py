# ruff: noqa: F821
import typing as t
from dask.distributed import Future
from superduperdb.core.job import ComponentJob
from superduperdb.core.serializable import Serializable
import dataclasses as dc

Datalayer = 'superduperdb.datalayer.base.datalayer.Datalayer'


@dc.dataclass
class Component(Serializable):
    """
    Base component which models, watchers, learning tasks etc. inherit from.

    :param identifier: Unique ID
    """

    variety: t.ClassVar[str]

    def _on_load(self, db):
        pass

    def _on_create(self, db):
        pass

    @property
    def child_components(self):
        return []

    @property
    def unique_id(self):
        if self.version is None:
            raise Exception('Version not yet set for component uniqueness')
        return f'{self.variety}/{self.identifier}/{self.version}'

    def dict(self):
        return dc.asdict(self)

    def create_validation_job(
        self,
        validation_set: t.Union[str, 'Dataset'],  # type: ignore[name-defined]
        metrics: t.List[str],
    ):
        return ComponentJob(
            component_identifier=self.identifier,
            method_name='predict',
            variety='model',
            kwargs={
                'distributed': False,
                'validation_set': validation_set,
                'metrics': metrics,
            },
        )

    def _validate(
        self,
        db: Datalayer,  # type: ignore[name-defined, valid-type]
        validation_set: t.Union[str, 'Dataset'],  # type: ignore[name-defined]
        metrics: t.List[t.Union['Metric', str]],  # type: ignore[name-defined]
    ):
        raise NotImplementedError

    def validate(
        self,
        db: Datalayer,  # type: ignore[name-defined, valid-type]
        validation_set: t.Union[str, 'Dataset'],  # type: ignore[name-defined]
        metrics: t.List[t.Union['Metric', str]],  # type: ignore[name-defined]
        distributed: bool = False,
        dependencies: t.Sequence[Future] = (),
    ):
        from .dataset import Dataset
        from .metric import Metric

        db.add(self)

        if isinstance(validation_set, Dataset):
            db.add(validation_set)
            validation_set = validation_set.identifier

        for i, m in enumerate(metrics):
            if isinstance(m, Metric):
                db.add(m)
                metrics[i] = m.identifier

        if distributed:
            return self.create_validation_job(
                validation_set=validation_set,
                metrics=metrics,
            )(db=db, distributed=True, dependencies=dependencies)

        output = self._validate(
            db=db,
            validation_set=validation_set,
            metrics=metrics,
        )
        if validation_set not in self.metric_values:
            self.metric_values[validation_set] = {}
        if self.metric_values[validation_set]:
            self.metric_values[validation_set].update(output)
        else:
            self.metric_values[validation_set] = output
        db.metadata.update_object(
            variety=self.variety,
            identifier=self.identifier,
            version=self.version,
            key='dict.metric_values',
            value=self.metric_values,
        )
        return self

    def schedule_jobs(self, database, dependencies=()):
        return []

    @classmethod
    def make_unique_id(cls, variety, identifier, version):
        return f'{variety}/{identifier}/{version}'
